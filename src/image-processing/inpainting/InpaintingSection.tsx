import { useMemo, useRef, useState, type PointerEvent } from 'react';
import inpaintCode from './inpaint.ts?raw';
import pythonCode from './inpaint.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { toImageData } from '../../lib/canvas';
import { scene } from '../../lib/demoImages';
import { diffusionInpaint, exemplarInpaint, holeError } from './inpaint';
import './inpainting.css';

const SIZE = 160;
const DISPLAY = 300;
const PATCH_RADIUS = 4;

type Preset = 'scratch' | 'box' | 'text';

interface Mask {
  width: number;
  height: number;
  data: Uint8Array;
}

function presetMask(preset: Preset, width: number, height: number): Mask {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = ctx.strokeStyle = '#fff';
    if (preset === 'scratch') {
      ctx.lineWidth = Math.max(4, width / 22);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(width * 0.08, height * 0.95);
      ctx.bezierCurveTo(width * 0.35, height * 0.55, width * 0.6, height * 0.75, width * 0.92, height * 0.3);
      ctx.stroke();
    } else if (preset === 'box') {
      ctx.fillRect(width * 0.58, height * 0.4, width * 0.2, height * 0.2);
    } else {
      ctx.font = `bold ${Math.round(height / 5)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('© 2026', width / 2, height * 0.62);
    }
  }
  const pixels = ctx?.getImageData(0, 0, width, height).data;
  return { width, height, data: Uint8Array.from({ length: width * height }, (_, i) => (pixels && pixels[i * 4 + 3] > 64 ? 1 : 0)) };
}

export function InpaintingSection() {
  const [source, setSource] = useState<ImageSource>(() => scene(256));
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [preset, setPreset] = useState<Preset>('scratch');
  const [painted, setPainted] = useState<Mask | null>(null);
  const [drawing, setDrawing] = useState<Mask | null>(null);
  const [brush, setBrush] = useState(4);
  const [iterations, setIterations] = useState(300);
  const [usePriority, setUsePriority] = useState(true);
  const [progress, setProgress] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, SIZE);
  const surface = useRef<HTMLDivElement>(null);

  // The mask in use: the painted one if it fits this image, otherwise the chosen preset.
  const mask = useMemo(() => {
    if (!image) return null;
    if (painted && painted.width === image.width && painted.height === image.height) return painted;
    return presetMask(preset, image.width, image.height);
  }, [image, painted, preset]);
  const shown = drawing ?? mask;

  const img = useMemo(() => (image ? { data: image.imageData.data, width: image.width, height: image.height } : null), [image]);
  const diffusion = useMemo(() => (img && mask ? diffusionInpaint(img, mask.data, iterations) : null), [img, mask, iterations]);
  const exemplar = useMemo(() => (img && mask ? exemplarInpaint(img, mask.data, PATCH_RADIUS, 40, usePriority) : null), [img, mask, usePriority]);

  const views = useMemo(() => {
    if (!img || !shown || !diffusion || !exemplar || !mask) return null;
    const damaged = new Uint8ClampedArray(img.data);
    shown.data.forEach((m, i) => { if (m) damaged.set([236, 72, 153], i * 4); });
    const lastStep = Math.max(1, ...exemplar.filledAt);
    const upTo = Math.round(progress * lastStep);
    const partial = new Uint8ClampedArray(exemplar.data);
    exemplar.filledAt.forEach((s, i) => { if (s > upTo) partial.set([236, 72, 153], i * 4); });
    return {
      damaged: toImageData(damaged, img.width, img.height),
      diffusion: toImageData(diffusion, img.width, img.height),
      exemplar: toImageData(partial, img.width, img.height),
      step: exemplar.steps[Math.min(exemplar.steps.length, upTo) - 1],
      diffusionError: holeError(img.data, diffusion, mask.data),
      exemplarError: holeError(img.data, exemplar.data, mask.data),
      holePixels: mask.data.reduce((s, v) => s + v, 0),
    };
  }, [img, shown, mask, diffusion, exemplar, progress]);

  const paint = (e: PointerEvent<HTMLDivElement>, current: Mask) => {
    const rect = surface.current?.getBoundingClientRect();
    if (!rect) return current;
    const cx = ((e.clientX - rect.left) / rect.width) * current.width, cy = ((e.clientY - rect.top) / rect.height) * current.height;
    const data = new Uint8Array(current.data);
    for (let y = Math.floor(cy - brush); y <= cy + brush; y++) for (let x = Math.floor(cx - brush); x <= cx + brush; x++)
      if (x >= 0 && y >= 0 && x < current.width && y < current.height && (x - cx) ** 2 + (y - cy) ** 2 <= brush * brush) data[y * current.width + x] = 1;
    return { ...current, data };
  };

  const error = uploadError ?? loadError;
  const box = (c: [number, number], cls: string) => (
    <rect x={c[0] - PATCH_RADIUS} y={c[1] - PATCH_RADIUS} width={2 * PATCH_RADIUS + 1} height={2 * PATCH_RADIUS + 1} className={cls} />
  );

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Damage:</span>
        {(['scratch', 'box', 'text'] as const).map(p => (
          <button key={p} type="button" aria-pressed={!painted && preset === p} className={`btn btn-sm${!painted && preset === p ? ' btn-primary' : ''}`}
            onClick={() => { setPreset(p); setPainted(null); setProgress(1); }}>
            {p === 'scratch' ? 'Scratch' : p === 'box' ? 'Missing block' : 'Watermark'}
          </button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => image && setPainted({ width: image.width, height: image.height, data: new Uint8Array(image.width * image.height) })}>
          Clear (paint your own)
        </button>
        <label className="dl-slider">Brush <strong>{brush} px</strong>
          <input type="range" min={1} max={10} value={brush} onChange={e => setBrush(Number(e.target.value))} />
        </label>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setUploadError(null); }} onError={setUploadError} />
        <span className="ml-auto">
          <ShowCodeButton code={inpaintCode} title="inpainting/inpaint.ts" python={{ code: pythonCode, title: 'inpainting/inpaint.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {views && image && (
        <div className="ip-grid">
          <div className="demo-panel">
            <h3>Damaged (paint on it)</h3>
            <PixelCanvas image={views.damaged} size={DISPLAY} label="The image with the hole to fill in pink; drag to paint more">
              <div
                ref={surface}
                className="ip-paint"
                onPointerDown={e => { if (!mask) return; e.currentTarget.setPointerCapture(e.pointerId); setDrawing(paint(e, mask)); }}
                onPointerMove={e => { if (drawing) setDrawing(paint(e, drawing)); }}
                onPointerUp={() => { if (drawing) { setPainted(drawing); setDrawing(null); setProgress(1); } }}
                onPointerCancel={() => setDrawing(null)}
              />
            </PixelCanvas>
            <p className="caption">{views.holePixels.toLocaleString()} missing pixels. Drag to paint more.</p>
          </div>
          <div className="demo-panel">
            <h3>Diffusion</h3>
            <PixelCanvas image={views.diffusion} size={DISPLAY} label="Hole filled by diffusion" />
            <label className="dl-slider ip-slider">Sweeps <strong>{iterations}</strong>
              <input type="range" min={0} max={1000} step={10} value={iterations} onChange={e => setIterations(Number(e.target.value))} />
            </label>
            <p className="caption">RMS error in the hole: {views.diffusionError.toFixed(1)}</p>
          </div>
          <div className="demo-panel">
            <h3>Exemplar-based (Criminisi)</h3>
            <PixelCanvas image={views.exemplar} size={DISPLAY} label="Hole filled by copying patches">
              {views.step && progress < 1 && (
                <svg className="ip-boxes" viewBox={`0 0 ${image.width} ${image.height}`} preserveAspectRatio="none" aria-hidden="true">
                  {box(views.step.source, 'ip-source')}
                  {box(views.step.target, 'ip-target')}
                </svg>
              )}
            </PixelCanvas>
            <label className="dl-slider ip-slider">Progress <strong>{Math.round(progress * 100)}%</strong>
              <input type="range" min={0} max={1} step={0.01} value={progress} onChange={e => setProgress(Number(e.target.value))} />
            </label>
            <label className="dl-slider ip-slider"><input type="checkbox" checked={usePriority} onChange={e => setUsePriority(e.target.checked)} /> Edges first (priority)</label>
            <p className="caption">RMS error in the hole: {views.exemplarError.toFixed(1)}{progress < 1 && <> · <span className="ip-key-source">source</span> → <span className="ip-key-target">target</span></>}</p>
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Filling holes</h4>
        <p>Removing a scratch, a watermark or an unwanted object leaves a hole, and the only information about what belongs there is
        what surrounds it. <strong>Diffusion</strong> makes each missing pixel the average of its four neighbours, sweep after sweep,
        until nothing changes: colour flows in from the border like heat. The result is as smooth as possible, which is right for sky
        and wrong for edges and texture: watch the stripes and the box edge turn to mush. Set the sweeps low to see the fill creep in.</p>
        <p className="formula"><code>u(x, y) ← ¼ · [u(x−1, y) + u(x+1, y) + u(x, y−1) + u(x, y+1)]</code></p>
        <p><strong>Exemplar-based</strong> inpainting copies instead of averaging. It takes a patch on the edge of the hole, searches the
        image for the known patch that best matches the part it can see, and pastes the rest. Drag the progress slider back to watch it
        patch by patch (pink box: the patch being filled; cyan: where it copies from). The <strong>order</strong> matters: Criminisi et al.
        fill first where many pixels are already known (confidence) and where an edge runs into the hole (data term), so lines get
        continued before flat areas close over them. Untick "edges first" to fill in plain scan order and compare.</p>
        <p>Modern tools (content-aware fill, diffusion-model inpainting) use the same two ideas, matching and propagation, with
        learned priors about what images look like.</p>
      </div>
    </div>
  );
}
