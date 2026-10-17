import { useMemo, useRef, useState, type PointerEvent } from 'react';
import tpsCode from './tps.ts?raw';
import pythonCode from './tps.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { createCanvas, imageDataToCanvas, toImageData } from '../../lib/canvas';
import { scene } from '../../lib/demoImages';
import { applyTps, bendingEnergy, fitTps, warpImage, type Point } from './tps';
import './warping.css';

const SIZE = 192;
const DISPLAY = 380;

const GRID: Point[] = [];
for (const y of [0.15, 0.5, 0.85]) for (const x of [0.15, 0.5, 0.85]) GRID.push({ x, y });

/** Named deformations: where each of the nine grid handles goes. */
const PRESETS: Record<string, (p: Point) => Point> = {
  // The bottom corners rise, the bottom centre stays.
  Smile: p => ({ x: p.x, y: p.y > 0.7 ? p.y - 0.25 * (p.x - 0.5) ** 2 : p.y }),
  Bulge: p => (p.x === 0.5 && p.y === 0.5 ? p : { x: 0.5 + (p.x - 0.5) * 1.12, y: 0.5 + (p.y - 0.5) * 1.12 }),
  Twist: p => {
    const a = 0.35 * (1 - Math.hypot(p.x - 0.5, p.y - 0.5) / 0.5), c = Math.cos(a), s = Math.sin(a);
    return { x: 0.5 + c * (p.x - 0.5) - s * (p.y - 0.5), y: 0.5 + s * (p.x - 0.5) + c * (p.y - 0.5) };
  },
  Shear: p => ({ x: p.x + 0.2 * (p.y - 0.5), y: p.y }),
};

interface Handle { src: Point; dst: Point }

const initialHandles = (): Handle[] => GRID.map(p => ({ src: p, dst: { ...p } }));

/** The image with a grid drawn over it, so the warp is visible everywhere. */
function withGrid(image: ImageData): Uint8ClampedArray {
  const { ctx } = createCanvas(image.width, image.height);
  ctx.drawImage(imageDataToCanvas(image), 0, 0);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.lineWidth = 1;
  const step = image.width / 12;
  ctx.beginPath();
  for (let v = step; v < image.width; v += step) { ctx.moveTo(v + 0.5, 0); ctx.lineTo(v + 0.5, image.height); }
  for (let v = step; v < image.height; v += step) { ctx.moveTo(0, v + 0.5); ctx.lineTo(image.width, v + 0.5); }
  ctx.stroke();
  return ctx.getImageData(0, 0, image.width, image.height).data;
}

export function WarpingSection() {
  const [source, setSource] = useState<ImageSource>(() => scene(SIZE));
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [handles, setHandles] = useState<Handle[]>(initialHandles);
  const [logLambda, setLogLambda] = useState(-6);
  const [showGrid, setShowGrid] = useState(true);
  const [showCode, setShowCode] = useState(false);
  const drag = useRef<number | null>(null);
  const { image, error: loadError } = useImageData(source, SIZE);
  const lambda = logLambda <= -6 ? 0 : 10 ** logLambda;

  const size = image ? Math.max(image.width, image.height) : SIZE;
  const pixels = useMemo(() => (image ? (showGrid ? withGrid(image.imageData) : image.imageData.data) : null), [image, showGrid]);
  const original = useMemo(() => (image && pixels ? toImageData(Uint8ClampedArray.from(pixels), image.width, image.height) : null), [image, pixels]);

  const result = useMemo(() => {
    if (!image || !pixels) return null;
    try {
      // Inverse map: from where each handle is now back to where it came from.
      const inverse = fitTps(handles.map(h => h.dst), handles.map(h => h.src), lambda);
      const forward = fitTps(handles.map(h => h.src), handles.map(h => h.dst), lambda);
      return {
        warped: toImageData(warpImage(pixels, image.width, image.height, inverse), image.width, image.height),
        energy: bendingEnergy(forward),
        // With λ > 0 a handle no longer lands exactly where it was dragged.
        landed: handles.map(h => applyTps(forward, h.src)),
        error: null,
      };
    } catch (e) {
      return { warped: null, energy: 0, landed: [], error: e instanceof Error ? e.message : 'Cannot fit the spline' };
    }
  }, [image, pixels, handles, lambda]);

  const toPoint = (e: PointerEvent<SVGSVGElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    const w = image?.width ?? SIZE, h = image?.height ?? SIZE;
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)) * (w / size),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)) * (h / size),
    };
  };

  const onPointerDown = (e: PointerEvent<SVGSVGElement>) => {
    const p = toPoint(e);
    const near = handles.findIndex(h => Math.hypot(h.dst.x - p.x, h.dst.y - p.y) < 0.04);
    e.currentTarget.setPointerCapture(e.pointerId);
    if (near >= 0) {
      drag.current = near;
    } else {
      // A new handle pins the image where it is.
      drag.current = handles.length;
      setHandles(hs => [...hs, { src: p, dst: { ...p } }]);
    }
  };
  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (drag.current === null) return;
    const p = toPoint(e), i = drag.current;
    setHandles(hs => hs.map((h, k) => (k === i ? { ...h, dst: p } : h)));
  };
  const onPointerUp = () => { drag.current = null; };

  const w = image?.width ?? SIZE, h = image?.height ?? SIZE;
  const error = uploadError ?? loadError ?? result?.error ?? null;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Deform:</span>
        {Object.entries(PRESETS).map(([name, f]) => (
          <button key={name} type="button" className="btn btn-sm" onClick={() => setHandles(GRID.map(p => ({ src: p, dst: f(p) })))}>{name}</button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => setHandles(initialHandles())}>↺ Reset</button>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(scene(SIZE)); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={tpsCode} title="warping/tps.ts" python={{ code: pythonCode, title: 'warping/tps.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Smoothing λ <strong>{lambda === 0 ? '0 (exact)' : lambda.toExponential(0)}</strong>
            <input type="range" min={-6} max={1} step={0.25} value={logLambda} onChange={e => setLogLambda(Number(e.target.value))} />
          </label>
          <label className="dl-slider"><input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} /> Grid</label>
        </div>
        {result && !result.error && <span className="dl-stat-row ml-auto">Bending energy <strong>{result.energy.toFixed(3)}</strong> · {handles.length} control points</span>}
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={original} size={DISPLAY * 0.6} pixelated={false} label="The original image" />
        </div>
        <div className="demo-panel">
          <h3>Warped · drag the handles, click to add one</h3>
          <PixelCanvas image={result?.warped ?? null} size={DISPLAY} pixelated={false} label="The image warped by the thin-plate spline; drag handles to deform it">
            <svg className="warp-overlay" viewBox={`0 0 ${w / size} ${h / size}`} preserveAspectRatio="none"
              onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
              role="application" aria-label="Control points">
              {handles.map((hd, i) => (
                <g key={i}>
                  <line className="warp-arrow" x1={hd.src.x} y1={hd.src.y} x2={hd.dst.x} y2={hd.dst.y} />
                  <circle className="warp-src" cx={hd.src.x} cy={hd.src.y} r={0.008} />
                  {result?.landed[i] && lambda > 0 && <circle className="warp-landed" cx={result.landed[i].x} cy={result.landed[i].y} r={0.012} />}
                  <circle className="warp-handle" cx={hd.dst.x} cy={hd.dst.y} r={0.016} />
                </g>
              ))}
            </svg>
          </PixelCanvas>
          <p className="caption">● handle · ∘ where it started{lambda > 0 ? ' · ◌ where the smoothed spline really puts it' : ''}</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>A warp that bends as little as possible</h4>
        <p>Drag a handle and the image follows: near the handle the image moves with it, far away it barely moves, and in
        between the motion blends smoothly. A <strong>thin-plate spline</strong> is the smoothest map that takes every control
        point exactly where you put it: the shape a thin metal plate takes when pinned at those points. Each coordinate is an
        affine map plus one radial bump per control point:</p>
        <p className="formula"><code>f(p) = a₀ + a₁x + a₂y + Σᵢ wᵢ · U(‖p − cᵢ‖),   U(r) = r² log r²</code></p>
        <p>The n weights and 3 affine terms come from one (n + 3)-unknown linear system, with the extra constraints that the
        weights sum to zero against 1, x and y, so the bumps add no affine motion of their own. If the handles move by an
        affine map (try <em>Shear</em>), all wᵢ are zero and the <strong>bending energy</strong> is 0.</p>
        <p>The <strong>smoothing λ</strong> adds λ to the diagonal: the spline may miss the handles a little to bend less, and as
        λ grows it tends to the best affine fit. The image itself is warped by <strong>inverse mapping</strong>: the spline
        fitted from the handles' new positions back to their old ones tells every output pixel where to fetch its colour,
        so there are no holes. Thin-plate splines are used to align medical scans, morph faces and correct document photos.</p>
      </div>
    </div>
  );
}
