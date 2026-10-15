import { useEffect, useMemo, useState } from 'react';
import flowCode from './opticalFlow.ts?raw';
import pythonCode from './optical_flow.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { grayToImageData } from '../../lib/image';
import { gridPoints, lucasKanade, makeFrames, type Motion } from './opticalFlow';
import './optical-flow.css';

const SIZE = 160;
const STEP = 12;
const DISPLAY = 340;
const ARROW_SCALE = 2;
/** Below this minimum eigenvalue (per pixel) the motion counts as ambiguous. */
const AMBIGUOUS = 0.5;

const AMOUNT: Record<Motion, { label: string; max: number; step: number; initial: number; unit: string }> = {
  translate: { label: 'Shift', max: 14, step: 0.5, initial: 3, unit: ' px' },
  rotate: { label: 'Angle', max: 10, step: 0.5, initial: 4, unit: '°' },
  zoom: { label: 'Zoom', max: 15, step: 0.5, initial: 6, unit: '%' },
};

export function OpticalFlowSection() {
  const [motion, setMotion] = useState<Motion>('translate');
  const [amount, setAmount] = useState(AMOUNT.translate.initial);
  const [pattern, setPattern] = useState<'texture' | 'stripes'>('texture');
  const [levels, setLevels] = useState(1);
  const [radius, setRadius] = useState(5);
  const [blink, setBlink] = useState(true);
  const [second, setSecond] = useState(false);
  const [showCode, setShowCode] = useState(false);

  useEffect(() => {
    if (!blink) return;
    const id = setInterval(() => setSecond(s => !s), 600);
    return () => clearInterval(id);
  }, [blink]);

  const frames = useMemo(() => makeFrames(motion, amount, SIZE, SIZE, pattern), [motion, amount, pattern]);
  const flow = useMemo(
    () => lucasKanade(frames.frame1, frames.frame2, gridPoints(SIZE, SIZE, STEP, 14), radius, levels),
    [frames, radius, levels],
  );
  const stats = useMemo(() => {
    const errors = flow.map(f => { const [u, v] = frames.truth(f.x, f.y); return Math.hypot(f.u - u, f.v - v); });
    return {
      errors,
      mean: errors.reduce((a, b) => a + b, 0) / errors.length,
      bad: errors.filter(e => e > 1).length / errors.length,
      ambiguous: flow.filter(f => f.minEigen < AMBIGUOUS).length,
    };
  }, [flow, frames]);

  const images = useMemo(() => ({
    a: grayToImageData(frames.frame1.pixels, SIZE, SIZE),
    b: grayToImageData(frames.frame2.pixels, SIZE, SIZE),
  }), [frames]);
  const a = AMOUNT[motion];
  const showingSecond = blink ? second : false;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Motion:</span>
        {(['translate', 'rotate', 'zoom'] as const).map(m => (
          <button key={m} type="button" aria-pressed={motion === m} className={`btn btn-sm${motion === m ? ' btn-primary' : ''}`}
            onClick={() => { setMotion(m); setAmount(AMOUNT[m].initial); }}>{m}</button>
        ))}
        <span className="demo-label of-gap">Pattern:</span>
        {(['texture', 'stripes'] as const).map(p => (
          <button key={p} type="button" aria-pressed={pattern === p} className={`btn btn-sm${pattern === p ? ' btn-primary' : ''}`}
            onClick={() => setPattern(p)}>{p}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={flowCode} title="optical-flow/opticalFlow.ts" python={{ code: pythonCode, title: 'optical-flow/optical_flow.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <label className="dl-slider">{a.label} <strong>{amount}{a.unit}</strong>
          <input type="range" min={0} max={a.max} step={a.step} value={amount} onChange={e => setAmount(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Window <strong>{2 * radius + 1}×{2 * radius + 1}</strong>
          <input type="range" min={1} max={10} value={radius} onChange={e => setRadius(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Pyramid levels <strong>{levels}</strong>
          <input type="range" min={1} max={4} value={levels} onChange={e => setLevels(Number(e.target.value))} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={blink} onChange={e => setBlink(e.target.checked)} /> Flick between the frames</label>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>{showingSecond ? 'Frame 2' : 'Frame 1'}</h3>
          <PixelCanvas image={showingSecond ? images.b : images.a} size={DISPLAY} label={showingSecond ? 'Second frame' : 'First frame'} />
          <p className="caption">{blink ? 'Flicking every 0.6 s: watch the pattern move.' : 'Tick "Flick" to see the motion.'}</p>
        </div>
        <div className="demo-panel">
          <h3>Lucas–Kanade estimate vs truth</h3>
          <PixelCanvas image={images.a} size={DISPLAY} label="Estimated and true motion vectors over the first frame">
            <svg className="of-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`}>
              <defs>
                <marker id="of-head" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                  <path d="M0,0 L6,3 L0,6 z" fill="context-stroke" />
                </marker>
              </defs>
              {flow.map((f, i) => {
                const [u, v] = frames.truth(f.x, f.y);
                const err = stats.errors[i];
                const ambiguous = f.minEigen < AMBIGUOUS;
                const colour = ambiguous ? '#9ca3af' : err < 0.3 ? '#34d399' : err < 1 ? '#facc15' : '#f87171';
                return (
                  <g key={i}>
                    <line x1={f.x} y1={f.y} x2={f.x + u * ARROW_SCALE} y2={f.y + v * ARROW_SCALE} className="of-truth" />
                    <line x1={f.x} y1={f.y} x2={f.x + f.u * ARROW_SCALE} y2={f.y + f.v * ARROW_SCALE} stroke={colour}
                      className={ambiguous ? 'of-arrow of-ambiguous' : 'of-arrow'} markerEnd="url(#of-head)" />
                  </g>
                );
              })}
            </svg>
          </PixelCanvas>
          <ul className="of-legend">
            <li><i className="of-key" style={{ background: '#34d399' }} />error &lt; 0.3 px</li>
            <li><i className="of-key" style={{ background: '#facc15' }} />&lt; 1 px</li>
            <li><i className="of-key" style={{ background: '#f87171' }} />wrong</li>
            <li><i className="of-key" style={{ background: '#9ca3af' }} />ambiguous</li>
            <li><i className="of-key of-key-truth" />true motion</li>
          </ul>
          <p className="caption">Arrows drawn {ARROW_SCALE}× longer than the motion.</p>
        </div>
      </div>

      <div className="dl-stat-row">
        <span>Mean error <strong>{stats.mean.toFixed(2)} px</strong></span>
        <span>Wrong by &gt; 1 px <strong>{(stats.bad * 100).toFixed(0)}%</strong></span>
        <span>Ambiguous <strong>{stats.ambiguous}/{flow.length}</strong></span>
      </div>

      <div className="demo-explanation">
        <h4>Lucas–Kanade optical flow</h4>
        <p>If a patch keeps its brightness while moving by (u, v), then for every pixel of a small window
        <code> Ix·u + Iy·v + It = 0</code>, where Ix, Iy are the image gradients and It the change between frames. One equation per pixel,
        two unknowns: least squares over the window gives a 2×2 system, solved independently at each arrow.</p>
        <p className="formula"><code>[ΣIx² ΣIxIy; ΣIxIy ΣIy²] · (u, v) = −(ΣIxIt, ΣIyIt)</code></p>
        <p>Try the <strong>stripes</strong>: moving along a stripe changes nothing in the image, so only the motion across the stripes can be measured.
        This is the <strong>aperture problem</strong>; the matrix is singular and the arrows turn gray. Then push the <strong>shift</strong> above
        about 7 px: the linear approximation only holds for small motions and the estimates go wrong, until you add <strong>pyramid levels</strong>.
        On a half-size image the motion is half as large; estimate it there, double it, and refine on the next finer level.</p>
        <p>Rotation and zoom move every pixel differently, but the method assumes one motion per window: a bigger window averages more
        (smoother, less noise-sensitive) but blurs the flow field. Deep networks such as RAFT have replaced Lucas–Kanade for dense flow,
        but the brightness-constancy idea is the same.</p>
      </div>
    </div>
  );
}
