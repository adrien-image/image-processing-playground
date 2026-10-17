import { useEffect, useMemo, useState } from 'react';
import flowCode from './opticalFlow.ts?raw';
import pythonCode from './optical_flow.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { grayToImageData } from '../../lib/image';
import { gridPoints, lucasKanade, makeFrames, type Motion } from './opticalFlow';
import { useText } from '../../lib/i18n';
import { TEXT } from './OpticalFlowSection.text';
import './optical-flow.css';

const SIZE = 160;
const STEP = 12;
const DISPLAY = 340;
const ARROW_SCALE = 2;
/** Below this minimum eigenvalue (per pixel) the motion counts as ambiguous. */
const AMBIGUOUS = 0.5;

const AMOUNT: Record<Motion, { max: number; step: number; initial: number; unit: string }> = {
  translate: { max: 14, step: 0.5, initial: 3, unit: ' px' },
  rotate: { max: 10, step: 0.5, initial: 4, unit: '°' },
  zoom: { max: 15, step: 0.5, initial: 6, unit: '%' },
};

export function OpticalFlowSection() {
  const t = useText(TEXT);
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
        <span className="demo-label">{t.motion}</span>
        {(['translate', 'rotate', 'zoom'] as const).map(m => (
          <button key={m} type="button" aria-pressed={motion === m} className={`btn btn-sm${motion === m ? ' btn-primary' : ''}`}
            onClick={() => { setMotion(m); setAmount(AMOUNT[m].initial); }}>{t.motions[m]}</button>
        ))}
        <span className="demo-label of-gap">{t.pattern}</span>
        {(['texture', 'stripes'] as const).map(p => (
          <button key={p} type="button" aria-pressed={pattern === p} className={`btn btn-sm${pattern === p ? ' btn-primary' : ''}`}
            onClick={() => setPattern(p)}>{t.patterns[p]}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={flowCode} title="optical-flow/opticalFlow.ts" python={{ code: pythonCode, title: 'optical-flow/optical_flow.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <label className="dl-slider">{t.amount[motion]} <strong>{amount}{a.unit}</strong>
          <input type="range" min={0} max={a.max} step={a.step} value={amount} onChange={e => setAmount(Number(e.target.value))} />
        </label>
        <label className="dl-slider">{t.window} <strong>{2 * radius + 1}×{2 * radius + 1}</strong>
          <input type="range" min={1} max={10} value={radius} onChange={e => setRadius(Number(e.target.value))} />
        </label>
        <label className="dl-slider">{t.levels} <strong>{levels}</strong>
          <input type="range" min={1} max={4} value={levels} onChange={e => setLevels(Number(e.target.value))} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={blink} onChange={e => setBlink(e.target.checked)} /> {t.flick}</label>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>{showingSecond ? t.frame2 : t.frame1}</h3>
          <PixelCanvas image={showingSecond ? images.b : images.a} size={DISPLAY} label={showingSecond ? t.secondFrame : t.firstFrame} />
          <p className="caption">{blink ? t.flicking : t.tickFlick}</p>
        </div>
        <div className="demo-panel">
          <h3>{t.estimate}</h3>
          <PixelCanvas image={images.a} size={DISPLAY} label={t.estimateLabel}>
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
            <li><i className="of-key" style={{ background: '#34d399' }} />{t.legendGood}</li>
            <li><i className="of-key" style={{ background: '#facc15' }} />{t.legendOk}</li>
            <li><i className="of-key" style={{ background: '#f87171' }} />{t.legendWrong}</li>
            <li><i className="of-key" style={{ background: '#9ca3af' }} />{t.legendAmbiguous}</li>
            <li><i className="of-key of-key-truth" />{t.legendTruth}</li>
          </ul>
          <p className="caption">{t.arrowsScale(ARROW_SCALE)}</p>
        </div>
      </div>

      <div className="dl-stat-row">
        <span>{t.meanError} <strong>{stats.mean.toFixed(2)} px</strong></span>
        <span>{t.wrongBy} <strong>{(stats.bad * 100).toFixed(0)}%</strong></span>
        <span>{t.ambiguous} <strong>{stats.ambiguous}/{flow.length}</strong></span>
      </div>

      <div className="demo-explanation">{t.explanation}</div>
    </div>
  );
}
