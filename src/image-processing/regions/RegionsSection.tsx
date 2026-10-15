import { memo, useCallback, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import componentsCode from './components.ts?raw';
import pythonCode from './components.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource, type LoadedImage } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { applyThreshold, histogram256, otsuThreshold } from '../segmentation/otsu';
import {
  labelColour, labelComponents, regionProperties, removeSmall,
  type Connectivity, type FilteredRegions, type Region,
} from './components';
import { followRegion, largestRegions, openMask, regionAt, regionPixels, strongestPixel } from './pipeline';
import './regions.css';

const DEMO_W = 200, DEMO_H = 160;

/**
 * Bright "cells" on a dark, unevenly lit background. Includes a touching pair (merged
 * into one region), two squares that meet only at a corner (4- vs 8-connectivity),
 * a ring with a hole, and a few 1–3 px specks for the clean-up step.
 */
function demoImage(): HTMLCanvasElement {
  return makeCanvas(DEMO_W, DEMO_H, ctx => {
    const bg = ctx.createRadialGradient(55, 40, 5, 100, 80, 200);
    bg.addColorStop(0, '#3c3c3c');
    bg.addColorStop(1, '#121212');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, DEMO_W, DEMO_H);

    // Faint sensor noise on the background (deterministic).
    const img = ctx.getImageData(0, 0, DEMO_W, DEMO_H);
    const random = seededRandom(7);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (random() + random() + random() - 1.5) * 10;
      for (let c = 0; c < 3; c++) img.data[i + c] = Math.max(0, Math.min(255, img.data[i + c] + n));
    }
    ctx.putImageData(img, 0, 0);

    // Each cell is a little brighter in the middle, and cells differ in brightness.
    const shade = (cx: number, cy: number, r: number, v: number) => {
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, `rgb(${v + 25}, ${v + 25}, ${v + 25})`);
      g.addColorStop(1, `rgb(${v - 20}, ${v - 20}, ${v - 20})`);
      ctx.fillStyle = g;
    };
    const disc = (cx: number, cy: number, r: number, v: number) => {
      shade(cx, cy, r, v);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    };
    const ellipse = (cx: number, cy: number, rx: number, ry: number, rot: number, v: number) => {
      shade(cx, cy, Math.max(rx, ry), v);
      ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, rot, 0, Math.PI * 2); ctx.fill();
    };

    disc(30, 28, 15, 210);
    disc(72, 20, 9, 185);
    ellipse(122, 26, 19, 9, 0.35, 200);
    disc(176, 18, 7, 165);
    shade(169, 58, 24, 200);
    ctx.beginPath(); ctx.roundRect(148, 44, 42, 28, 9); ctx.fill();
    disc(76, 46, 5, 175);
    // A touching pair: one region for the labelling.
    disc(88, 76, 14, 210);
    disc(110, 82, 12, 190);
    // A ring: one region with a hole (even–odd fill leaves the inner disc empty).
    shade(36, 84, 18, 205);
    ctx.beginPath(); ctx.arc(36, 84, 18, 0, Math.PI * 2); ctx.arc(36, 84, 10, 0, Math.PI * 2); ctx.fill('evenodd');
    disc(96, 112, 4, 170);
    ellipse(18, 134, 7, 14, 0, 190);
    disc(58, 132, 11, 220);
    ellipse(118, 138, 13, 7, -0.5, 195);
    disc(152, 140, 8, 180);
    disc(184, 142, 9, 205);

    // Two squares that touch only at one corner, at pixels (163, 105) and (164, 106).
    ctx.fillStyle = '#e1e1e1';
    ctx.fillRect(150, 92, 14, 14);
    ctx.fillRect(164, 106, 14, 14);

    // Bright specks of 1–3 px.
    ctx.fillStyle = '#dcdcdc';
    ctx.fillRect(60, 100, 1, 1);
    ctx.fillRect(134, 58, 2, 2);
    ctx.fillRect(192, 96, 1, 2);
    ctx.fillRect(100, 150, 3, 3);
    ctx.fillRect(6, 58, 2, 1);
    ctx.fillRect(128, 110, 1, 1);
  });
}

type OpeningSize = 0 | 3 | 5;
const OPENINGS: { size: OpeningSize; label: string }[] = [
  { size: 0, label: 'Off' },
  { size: 3, label: '3×3' },
  { size: 5, label: '5×5' },
];

const DISPLAY = 300;
const MAX_MIN_AREA = 300;
/** Below this area, circularity is dominated by pixelation and shown dimmed. */
const TINY_AREA = 20;
/** At most this many regions get a table row and an outline (the largest ones). */
const MAX_LISTED = 300;
/** A click this close to a region (in CSS pixels) still selects it, so 1-px specks can be hit. */
const HIT_TOLERANCE_PX = 4;

type RGB = readonly [number, number, number];
const BLACK: RGB = [0, 0, 0];
const WHITE: RGB = [255, 255, 255];
const OPENED_AWAY: RGB = [220, 60, 60];
const DROPPED: RGB = [85, 85, 85];

/**
 * The selected region, remembered by its strongest pixel (and, as a fallback, all its
 * pixels) rather than by its label, so the selection follows the region while the
 * threshold, opening, connectivity or minimum area change (labels get renumbered).
 * Valid only for the image it was made on.
 */
interface Selection {
  image: LoadedImage;
  pixel: number;
  pixels: Int32Array;
}

/** The hovered or focused row. Valid only for the labelling it came from. */
interface Hover {
  of: FilteredRegions;
  label: number;
}

/** Overlay sizes in image pixels, chosen to give the same CSS size for any image size. */
interface OverlaySizes {
  font: number;
  cross: number;
  outline: number;
  gap: number;
}

function overlaySizes(width: number, height: number): OverlaySizes {
  const scale = Math.max(width, height) / DISPLAY; // image pixels per CSS pixel at full size
  return { font: 12 * scale, cross: 4 * scale, outline: 3 * scale, gap: 1.5 * scale };
}

const fmt = (v: number, digits = 1) => v.toFixed(digits);

/** Bounding box, centroid cross and (when there is room, or when active) the label number. */
function regionMark(r: Region, s: OverlaySizes, active: boolean) {
  const { x0, y0, x1, y1 } = r.bbox;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const showNumber = active || (bw >= s.font * 1.3 && bh >= s.font * 1.3);
  // Number just above the box, like a detector's label; inside the box at the top edge.
  const above = y0 >= s.font;
  const { x, y } = r.centroid;
  const cross = `M${x - s.cross} ${y}H${x + s.cross}M${x} ${y - s.cross}V${y + s.cross}`;
  return (
    <g key={r.label} className={active ? 'regions-mark-active' : undefined}>
      <rect x={x0} y={y0} width={bw} height={bh} className="regions-bbox" />
      <path d={cross} className="regions-centroid-halo" />
      <path d={cross} className="regions-centroid" />
      {showNumber && (
        <text x={x0} y={above ? y0 - s.gap : y0 + s.gap} fontSize={s.font} strokeWidth={s.outline}
          className={`regions-num${above ? ' regions-num-above' : ''}`}>
          {r.label}
        </text>
      )}
    </g>
  );
}

/** The whole image minus the region's box: filled translucent black, it dims everything else. */
function dimOutside({ bbox: { x0, y0, x1, y1 } }: Region, width: number, height: number): string {
  return `M0 0H${width}V${height}H0Z M${x0} ${y0}V${y1 + 1}H${x1 + 1}V${y0}Z`;
}

/** Scroll the table (not the page) so the row of `label` is visible below the sticky header. */
function revealRow(wrap: HTMLElement | null, label: number) {
  const row = wrap?.querySelector<HTMLElement>(`tr[data-label="${label}"]`);
  if (!wrap || !row) return;
  const head = wrap.querySelector('thead')?.getBoundingClientRect().height ?? 0;
  const box = wrap.getBoundingClientRect();
  const r = row.getBoundingClientRect();
  if (r.top < box.top + head) wrap.scrollTop -= box.top + head - r.top;
  else if (r.bottom > box.bottom) wrap.scrollTop += r.bottom - box.bottom;
}

interface RegionRowProps {
  region: Region;
  active: boolean;
  selected: boolean;
  onHover: (label: number | null) => void;
  onToggle: (label: number, selected: boolean) => void;
}

/** One table row. Memoised, so hovering re-renders only the rows whose state changes. */
const RegionRow = memo(function RegionRow({ region: r, active, selected, onHover, onToggle }: RegionRowProps) {
  const [cr, cg, cb] = labelColour(r.label);
  return (
    <tr data-label={r.label}
      className={`${active ? 'regions-row-active' : ''}${selected ? ' regions-row-selected' : ''}`}
      onMouseEnter={() => onHover(r.label)} onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(r.label)} onBlur={() => onHover(null)}>
      <th scope="row">
        <button type="button" className="regions-row-btn" aria-pressed={selected}
          aria-label={`Region ${r.label}`} onClick={() => onToggle(r.label, selected)}>
          <span className="regions-swatch" style={{ background: `rgb(${cr}, ${cg}, ${cb})` }} />
          {r.label}
        </button>
      </th>
      <td>{r.area}</td>
      <td>{fmt(r.centroid.x)}, {fmt(r.centroid.y)}</td>
      <td>{r.bbox.x1 - r.bbox.x0 + 1}×{r.bbox.y1 - r.bbox.y0 + 1}</td>
      <td>{r.perimeter}</td>
      <td className={r.area < TINY_AREA ? 'muted' : undefined}>{fmt(r.circularity, 2)}</td>
      <td>{r.meanIntensity === null ? '—' : fmt(r.meanIntensity, 0)}</td>
    </tr>
  );
});

const describeRegion = (r: Region) =>
  `Region ${r.label}: area ${r.area} px, centroid (${fmt(r.centroid.x)}, ${fmt(r.centroid.y)}), `
  + `box ${r.bbox.x1 - r.bbox.x0 + 1}×${r.bbox.y1 - r.bbox.y0 + 1}, circularity ${fmt(r.circularity, 2)}`;

export function RegionsSection() {
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // null = follow Otsu's suggestion; a number = the user moved the slider.
  const [userThreshold, setUserThreshold] = useState<number | null>(null);
  const [dark, setDark] = useState(false);
  const [opening, setOpening] = useState<OpeningSize>(3);
  const [connectivity, setConnectivity] = useState<Connectivity>(8);
  const [minArea, setMinArea] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Hover | null>(null);
  const tableWrapRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<HTMLDivElement>(null);

  const { image, error: loadError, loading } = useImageData(source, 256);

  // 1. The values that get thresholded (inverted for dark objects) and Otsu's threshold.
  const prepared = useMemo(() => {
    if (!image) return null;
    const values = dark ? image.gray.map(v => 255 - v) : image.gray;
    return { values, otsu: otsuThreshold(histogram256(values)) };
  }, [image, dark]);

  const threshold = userThreshold ?? prepared?.otsu ?? 128;

  // 2. Threshold, then (optionally) open to remove specks.
  const masks = useMemo(() => {
    if (!image || !prepared) return null;
    const raw = applyThreshold(prepared.values, threshold);
    return { raw, clean: opening ? openMask(raw, image.width, image.height, opening) : raw };
  }, [image, prepared, threshold, opening]);

  // 3. Label and measure every region.
  const labelled = useMemo(() => {
    if (!image || !masks) return null;
    const { width, height, gray } = image;
    const { labels, count } = labelComponents(masks.clean, width, height, connectivity);
    return { labels, regions: regionProperties(labels, count, width, height, gray) };
  }, [image, masks, connectivity]);

  // 4. Drop regions below the minimum area.
  const result = useMemo(
    () => (labelled ? removeSmall(labelled.labels, labelled.regions, minArea) : null),
    [labelled, minArea],
  );

  // 5. What gets drawn. None of it depends on hover or selection, so those stay cheap.
  const listed = useMemo(() => (result ? largestRegions(result.regions, MAX_LISTED) : []), [result]);
  const w = image?.width ?? DEMO_W, h = image?.height ?? DEMO_H;
  const sizes = useMemo(() => overlaySizes(w, h), [w, h]);
  const marks = useMemo(() => listed.map(r => regionMark(r, sizes, false)), [listed, sizes]);

  const maskImage = useMemo(() => {
    if (!image || !masks) return null;
    const { raw, clean } = masks;
    return rgbImage(image.width, image.height, i => (clean[i] ? WHITE : raw[i] ? OPENED_AWAY : BLACK));
  }, [image, masks]);

  const labelImage = useMemo(() => {
    if (!image || !labelled || !result) return null;
    const palette = Array.from({ length: result.count + 1 }, (_, l) => labelColour(l));
    return rgbImage(image.width, image.height, i => {
      const l = result.labels[i];
      if (l === 0) return labelled.labels[i] ? DROPPED : BLACK;
      return palette[l];
    });
  }, [image, labelled, result]);

  const regions = result?.regions ?? [];
  const count = result?.count ?? 0;
  const dropped = labelled && result ? labelled.regions.length - result.count : 0;
  const selectionValid = selection !== null && selection.image === image && result !== null;
  const selectedLabel = useMemo(
    () => (selection && selection.image === image && result ? followRegion(result.labels, selection.pixel, selection.pixels) : null),
    [selection, image, result],
  );
  const selectedRegion = selectedLabel !== null ? regions[selectedLabel - 1] : null;
  // A selected region outside the listed ones still gets its row.
  const rows = selectedRegion && !listed.includes(selectedRegion)
    ? [...listed, selectedRegion].sort((a, b) => a.label - b.label)
    : listed;
  // A row that unmounted while hovered or focused (an unlisted region was deselected) never
  // reports the mouse leaving: ignore its hover.
  const hoveredLabel = hovered && hovered.of === result && rows.some(r => r.label === hovered.label) ? hovered.label : null;
  const activeLabel = hoveredLabel ?? selectedLabel;
  const active = activeLabel !== null ? regions[activeLabel - 1] : null;
  const error = uploadError ?? loadError;

  const hover = useCallback(
    (label: number | null) => setHovered(result && label !== null ? { of: result, label } : null),
    [result],
  );

  const selectLabel = useCallback((label: number | null) => {
    const region = label !== null ? result?.regions[label - 1] : undefined;
    if (!image || !prepared || !result || !region) {
      setSelection(null);
      return;
    }
    setSelection({
      image,
      pixel: strongestPixel(result.labels, prepared.values, image.width, region),
      pixels: regionPixels(result.labels, image.width, region),
    });
    // Next frame: the row exists by then, even if it was not listed before.
    requestAnimationFrame(() => revealRow(tableWrapRef.current, region.label));
  }, [image, prepared, result]);

  const toggleRow = useCallback((label: number, isSelected: boolean) => {
    // Deselecting a region outside the listed ones removes its row, and the focused button with
    // it: move the focus to the label map instead of letting it fall back to the page.
    if (isSelected && !listed.some(r => r.label === label)) mapRef.current?.focus();
    selectLabel(isSelected ? null : label);
  }, [listed, selectLabel]);

  const nextLabel = count > 0 ? ((selectedLabel ?? 0) % count) + 1 : null;

  const onMapClick = (e: MouseEvent<HTMLDivElement>) => {
    if (!result) return;
    // detail 0: a click made by assistive technology or element.click(), not a pointer. It has
    // no meaningful position, so treat it like Enter: select the next region.
    if (e.detail === 0) {
      selectLabel(nextLabel);
      return;
    }
    const box = (e.currentTarget.querySelector('.pixel-canvas') ?? e.currentTarget).getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * w;
    const y = ((e.clientY - box.top) / box.height) * h;
    if (!(x >= 0 && y >= 0 && x < w && y < h)) return; // outside the image: ignore
    selectLabel(regionAt(result.labels, w, h, x, y, HIT_TOLERANCE_PX * (w / box.width)));
  };

  const onMapKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (count === 0 && e.key !== 'Escape') return;
    const from = selectedLabel ?? 0;
    let target: number | null;
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': case 'Enter': case ' ':
        target = nextLabel;
        break;
      case 'ArrowLeft': case 'ArrowUp':
        target = from <= 1 ? count : from - 1;
        break;
      case 'Home':
        target = 1;
        break;
      case 'End':
        target = count;
        break;
      case 'Escape':
        target = null;
        break;
      default:
        return;
    }
    e.preventDefault();
    selectLabel(target);
  };

  const resetToDemo = () => {
    setSource(demoImage());
    setIsDemo(true);
    setUserThreshold(null);
    setUploadError(null);
  };

  const capped = count > listed.length;
  // Why there are no regions: say which step removed the last of them.
  let emptyHint = '';
  if (result && count === 0) {
    if (dropped > 0) {
      emptyHint = `No regions — ${dropped === 1 ? 'the only one is' : `all ${dropped} are`} smaller than ${minArea} px: lower the minimum area.`;
    } else if (opening && masks?.raw.includes(1)) {
      emptyHint = `No regions — the ${opening}×${opening} opening removed all the foreground: use a smaller opening or turn it off.`;
    } else {
      emptyHint = threshold > 0
        ? 'No regions — no pixel is above the threshold: lower it, or toggle “Objects are dark”.'
        : 'No regions — no pixel is above 0: toggle “Objects are dark”.';
    }
  }
  const noImageText = loading ? 'Loading image…' : 'No image loaded. Upload another file, or go back to the demo image.';

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <ImageUploader
          label="📁 Upload"
          onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); setUserThreshold(null); }}
          onError={setUploadError}
        />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={resetToDemo}>Demo image</button>}
        <button type="button" aria-pressed={dark} className={`btn btn-sm${dark ? ' btn-primary' : ''}`}
          onClick={() => { setDark(d => !d); setUserThreshold(null); }}>
          🌑 Objects are dark
        </button>
        <span className="regions-group">
          <span className="demo-label">Opening:</span>
          {OPENINGS.map(o => (
            <button key={o.size} type="button" aria-pressed={opening === o.size}
              aria-label={o.size ? `Opening ${o.label}` : 'Opening off'}
              className={`btn btn-sm${opening === o.size ? ' btn-primary' : ''}`} onClick={() => setOpening(o.size)}>
              {o.label}
            </button>
          ))}
        </span>
        <span className="regions-group">
          <span className="demo-label">Connectivity:</span>
          {([4, 8] as const).map(c => (
            <button key={c} type="button" aria-pressed={connectivity === c}
              className={`btn btn-sm${connectivity === c ? ' btn-primary' : ''}`} onClick={() => setConnectivity(c)}>
              {c}-connected
            </button>
          ))}
        </span>
        <span className="ml-auto">
          <ShowCodeButton code={componentsCode} title="regions/components.ts" python={{ code: pythonCode, title: 'regions/components.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="regions-threshold">
            Threshold: <strong>{prepared ? threshold : '—'}</strong>{' '}
            <span className="muted">(value &gt; t is foreground{dark ? '; value = 255 − gray' : ''})</span>
          </label>
          <input id="regions-threshold" type="range" min={0} max={254} step={1} value={threshold}
            disabled={!prepared} onChange={e => setUserThreshold(Number(e.target.value))} className="demo-slider" />
          {prepared && (
            <button type="button" className="regions-otsu" onClick={() => setUserThreshold(null)}
              aria-label={`Reset threshold to Otsu's value, ${prepared.otsu}`}>
              ⬤ Otsu: {prepared.otsu}
            </button>
          )}
        </div>
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="regions-min-area">
            Minimum area: <strong>{minArea}</strong> px <span className="muted">(smaller regions are dropped)</span>
          </label>
          <input id="regions-min-area" type="range" min={1} max={MAX_MIN_AREA} step={1} value={minArea}
            onChange={e => setMinArea(Number(e.target.value))} className="demo-slider" />
        </div>
      </div>

      <div className="panel regions-summary">
        <p className="regions-count" role="status">
          {result
            ? <><strong>{count}</strong> {count === 1 ? 'region' : 'regions'}</>
            : <><strong>—</strong> {loading ? 'loading…' : 'no image'}</>}
        </p>
        <p className="muted">
          {result ? (
            <>
              {connectivity}-connected · threshold {threshold} · opening {opening ? `${opening}×${opening}` : 'off'}
              {dropped > 0 && <> · {dropped} smaller than {minArea} px dropped</>}
            </>
          ) : loading ? 'Measuring starts when the image has loaded.' : 'Upload another file, or go back to the demo image.'}
        </p>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Original image"
            placeholderAspect={DEMO_W / DEMO_H}>
            {active && (
              <svg className="regions-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                <rect x={active.bbox.x0} y={active.bbox.y0} width={active.bbox.x1 - active.bbox.x0 + 1}
                  height={active.bbox.y1 - active.bbox.y0 + 1} className="regions-bbox regions-bbox-original" />
              </svg>
            )}
          </PixelCanvas>
          <p className="caption">{active ? `Box of region ${active.label}` : 'Input image'}</p>
        </div>
        <div className="demo-panel">
          <h3>Binary mask{opening ? ` + opening ${opening}×${opening}` : ''}</h3>
          <PixelCanvas image={maskImage} size={DISPLAY} label={`Binary mask, threshold ${threshold}`}
            placeholderAspect={DEMO_W / DEMO_H} />
          <p className="caption">
            White: foreground{opening ? <> · <span className="regions-key-removed">red</span>: removed by the opening</> : null}
          </p>
        </div>
        <div className="demo-panel">
          <h3>Labels</h3>
          <div
            ref={mapRef}
            className="regions-map"
            role="button"
            tabIndex={result ? 0 : -1}
            aria-disabled={!result}
            aria-label={'Label map. Click a region to select it and its row in the table. Keyboard: arrow keys, '
              + 'Enter or Space select the next or previous region, Home and End the first and last, '
              + 'Escape clears the selection.'}
            onClick={onMapClick}
            onKeyDown={onMapKey}
          >
            <PixelCanvas image={labelImage} size={DISPLAY} label={`Label map with ${count} coloured regions`}
              placeholderAspect={DEMO_W / DEMO_H}>
              <svg className="regions-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                {marks}
                {active && <path className="regions-dim" fillRule="evenodd" d={dimOutside(active, w, h)} />}
                {active && regionMark(active, sizes, true)}
              </svg>
            </PixelCanvas>
          </div>
          <p className="caption">
            One colour per region · □ bounding box · + centroid
            {capped ? ` (drawn for the ${listed.length} largest)` : ''}
            {dropped > 0 ? ' · gray: below the minimum area' : ''}
          </p>
        </div>
      </div>

      <p className="regions-selection" aria-live="polite">
        {selectedRegion ? describeRegion(selectedRegion)
          : selectionValid ? 'The selected region has gone at these settings: change them back to see it again, or pick another region.'
          : count > 0 ? 'Click a region on the label map, or a label in the table, to select it (or focus the map and use the arrow keys).'
          : ''}
      </p>

      <div className="regions-bottom">
        <div className="panel regions-table-panel">
          <h3 className="panel-heading">Measurements{result ? ` · ${count} ${count === 1 ? 'region' : 'regions'}` : ''}</h3>
          {capped && (
            <p className="regions-cap-note">
              Showing the {listed.length} largest of {count} regions. Raise the minimum area or turn on the opening to
              list them all.
            </p>
          )}
          <div className="regions-table-wrap" ref={tableWrapRef} role="region" aria-label="Region measurements" tabIndex={0}>
            <table className="regions-table">
              <thead>
                <tr>
                  <th scope="col">Label</th>
                  <th scope="col">Area (px)</th>
                  <th scope="col">Centroid (x, y)</th>
                  <th scope="col">Box w×h</th>
                  <th scope="col">Perimeter (px)</th>
                  <th scope="col">Circularity</th>
                  <th scope="col">Mean gray</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <RegionRow key={r.label} region={r} active={r.label === activeLabel} selected={r.label === selectedLabel}
                    onHover={hover} onToggle={toggleRow} />
                ))}
                {(!result || count === 0) && (
                  <tr>
                    <td colSpan={7} className="muted">
                      {result ? emptyHint : noImageText}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="caption">
            Hover or focus a row to highlight its region; click its label to select it. A selection follows its region
            while you change the settings. Circularity is dimmed for regions under {TINY_AREA} px, where pixelation
            dominates.
          </p>
        </div>

        <div className="demo-explanation">
          <h4>Connected-component labelling</h4>
          <p>Thresholding says <em>which</em> pixels are foreground, not which belong together. Labelling gives every
          connected blob its own number. This implementation scans the mask row by row; at each foreground pixel that has
          no label yet it starts a new region and <strong>flood-fills</strong> it with an explicit stack: pop a pixel,
          label and push its unlabelled foreground neighbours, repeat until the stack is empty. Every pixel is labelled
          once, so the cost grows linearly with the image, and regions are numbered in the order their top-most
          (then left-most) pixel is reached.</p>

          <h4>4- vs 8-connectivity</h4>
          <p>With <strong>4-connectivity</strong> two pixels are neighbours only if they share an edge; with
          <strong> 8-connectivity</strong> touching corners count too. In the demo, the two squares on the right meet at a
          single corner: they are two regions when 4-connected and one when 8-connected.</p>

          <h4>Clean up before counting</h4>
          <p>Every speck that survives thresholding is counted as a region. An <strong>opening</strong> (erosion, then
          dilation — see Morphology) with a k×k square deletes any blob that cannot hold a whole k×k square, and gives back
          the shape of larger blobs apart from their thinnest details. (Pixels beyond the image edge are ignored, so blobs
          touching the edge are treated a little more leniently.) In the demo, 3×3 removes the 1–2 px specks but
          not the 3×3 one, which needs 5×5 — or the <strong>minimum area</strong> filter, which drops regions with fewer
          pixels after labelling.</p>

          <h4>Touching objects merge</h4>
          <p>Labelling only follows connections, so objects that touch become one region: the touching pair of cells in
          the middle of the demo is counted once, and its area and centroid describe neither cell. Separating them needs
          shape information, typically the <strong>distance transform</strong> (each foreground pixel's distance to the
          background, which peaks once per cell) followed by a <strong>watershed</strong> that splits the region along the
          narrow neck between the peaks. Neither is done in this section.</p>
          <p>Holes do not split a region: the ring is one region whose hole stays background. Its centroid lies in the
          hole — a centroid is an average position, not necessarily a pixel of the region.</p>

          <h4>Measurements</h4>
          <p><strong>Area</strong> A is the pixel count. The <strong>centroid</strong> is the mean of the pixel centres
          (x + 0.5, y + 0.5), in pixels from the image's top-left corner. The <strong>box</strong> is the smallest
          axis-aligned rectangle containing the region. The <strong>perimeter</strong> P is the number of boundary
          pixels: region pixels with at least one 4-neighbour (left, right, up or down) that is background or outside
          the image. <strong>Mean gray</strong> is the average of the original gray image (0–255) over the region,
          also when “Objects are dark” is on.</p>
          <div className="formula"><code>circularity = 4πA / P²</code></div>
          <p>Circularity is 1 for a perfect circle and lower for elongated or ragged shapes, but on a pixel grid it is
          only approximate: P counts a diagonal step as one pixel instead of √2, so a large digital disk scores about 1.25
          and a large square tends to π/4 ≈ 0.79 (a 10×10 square still scores ≈ 0.97). Small regions score higher
          still (a single pixel: 4π ≈ 12.6). Use it to compare shapes of similar size, not as an absolute value.</p>
        </div>
      </div>
    </div>
  );
}
