import { useEffect, useMemo, useState } from 'react';
import detectionCode from './detection.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { canvasToImageData } from '../../lib/canvas';
import { makeCanvas } from '../../lib/demoImages';
import { seededRandom } from '../../lib/random';
import {
  anchors, evaluate, iou, labelAnchors, makeScene, nonMaxSuppression, simulateDetections,
  type Box, type Detection, type Evaluation, type SceneObject,
} from './detection';
import '../../components/LineChart.css';
import '../shared/dl-shared.css';
import './detection.css';

const SIZE = 256;
const DISPLAY = 480;

type Tab = 'nms' | 'pr' | 'anchors';

function drawScene(objects: SceneObject[], seed: number): ImageData {
  const random = seededRandom(seed * 7 + 1);
  return canvasToImageData(makeCanvas(SIZE, SIZE, ctx => {
    const bg = ctx.createLinearGradient(0, 0, SIZE, SIZE);
    bg.addColorStop(0, '#1d2433');
    bg.addColorStop(1, '#39425a');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, SIZE, SIZE);
    // A little clutter, so the "image" isn't trivially clean.
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(255,255,255,${0.03 + random() * 0.05})`;
      ctx.fillRect(random() * SIZE, random() * SIZE, 2 + random() * 10, 2 + random() * 10);
    }
    const colours = { square: '#f59e0b', disc: '#38bdf8', triangle: '#f472b6' };
    for (const o of objects) {
      ctx.fillStyle = colours[o.kind];
      const w = o.x1 - o.x0, h = o.y1 - o.y0;
      ctx.beginPath();
      if (o.kind === 'square') ctx.rect(o.x0, o.y0, w, h);
      else if (o.kind === 'disc') ctx.ellipse(o.x0 + w / 2, o.y0 + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
      else { ctx.moveTo(o.x0 + w / 2, o.y0); ctx.lineTo(o.x1, o.y1); ctx.lineTo(o.x0, o.y1); ctx.closePath(); }
      ctx.fill();
    }
  }));
}

export function DetectionSection() {
  const [tab, setTab] = useState<Tab>('nms');
  const [seed, setSeed] = useState(4);
  const [showCode, setShowCode] = useState(false);

  const { truth, dets, image } = useMemo(() => {
    const random = seededRandom(seed);
    const objects = makeScene(SIZE, 4 + (seed % 3), random);
    return { truth: objects, dets: simulateDetections(objects, SIZE, random), image: drawScene(objects, seed) };
  }, [seed]);

  return (
    <div className="demo-section">
      <div className="tab-bar">
        <div className="tab-list" role="tablist" aria-label="Detection views">
          {([['nms', 'Non-max suppression'], ['pr', 'Precision & recall'], ['anchors', 'Anchors']] as const).map(([key, label]) => (
            <button key={key} type="button" role="tab" aria-selected={tab === key}
              className={`tab-btn${tab === key ? ' tab-active' : ''}`} onClick={() => setTab(key)}>{label}</button>
          ))}
        </div>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 New scene</button>
        <ShowCodeButton code={detectionCode} title="detection/detection.ts" open={showCode} onOpenChange={setShowCode} />
      </div>

      {tab === 'nms' && <NmsView key={seed} image={image} truth={truth} dets={dets} />}
      {tab === 'pr' && <PrView image={image} truth={truth} dets={dets} />}
      {tab === 'anchors' && <AnchorView image={image} truth={truth} />}
    </div>
  );
}

// ---------------------------------------------------------------- NMS

function NmsView({ image, truth, dets }: { image: ImageData; truth: Box[]; dets: Detection[] }) {
  const [scoreThreshold, setScoreThreshold] = useState(0.3);
  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [applied, setApplied] = useState<number | null>(null); // null = all steps
  const [playing, setPlaying] = useState(false);
  const [showTruth, setShowTruth] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  const steps = useMemo(() => nonMaxSuppression(dets, iouThreshold, scoreThreshold), [dets, iouThreshold, scoreThreshold]);
  const done = applied === null ? steps.length : Math.min(applied, steps.length);

  useEffect(() => {
    if (!playing) return;
    if (done >= steps.length) { const t = setTimeout(() => setPlaying(false), 0); return () => clearTimeout(t); }
    const t = setTimeout(() => setApplied(done + 1), 700);
    return () => clearTimeout(t);
  }, [playing, done, steps.length]);

  const state = new Map<number, 'kept' | 'suppressed' | 'pending' | 'below'>();
  dets.forEach((d, i) => state.set(i, d.score < scoreThreshold ? 'below' : 'pending'));
  steps.slice(0, done).forEach(s => { state.set(s.kept, 'kept'); s.suppressed.forEach(i => state.set(i, 'suppressed')); });
  const current = done > 0 ? steps[done - 1] : null;
  const keptCount = steps.slice(0, done).length;

  return (
    <>
      <div className="demo-controls">
        <label className="dl-slider">Score threshold <strong>{scoreThreshold.toFixed(2)}</strong>
          <input type="range" min={0} max={0.9} step={0.05} value={scoreThreshold} onChange={e => { setScoreThreshold(Number(e.target.value)); setApplied(null); }} />
        </label>
        <label className="dl-slider">IoU threshold <strong>{iouThreshold.toFixed(2)}</strong>
          <input type="range" min={0.1} max={0.95} step={0.05} value={iouThreshold} onChange={e => { setIouThreshold(Number(e.target.value)); setApplied(null); }} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={showTruth} onChange={e => setShowTruth(e.target.checked)} /> Ground truth</label>
        <span className="ml-auto det-step-controls">
          <button type="button" className="btn btn-sm" onClick={() => { setApplied(0); setPlaying(false); }}>⏮ Raw output</button>
          <button type="button" className="btn btn-sm" onClick={() => { if (done >= steps.length) setApplied(0); setPlaying(p => !p); }}>
            {playing ? '⏸ Pause' : '▶ Run NMS'}
          </button>
          <button type="button" className="btn btn-sm" disabled={done >= steps.length} onClick={() => setApplied(done + 1)}>Step</button>
          <button type="button" className="btn btn-sm" onClick={() => { setApplied(null); setPlaying(false); }}>⏭ Result</button>
        </span>
      </div>

      <div className="det-layout">
        <div className="demo-panel">
          <PixelCanvas image={image} size={DISPLAY} pixelated={false} label="Scene with detected boxes">
            <svg className="det-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`}>
              {showTruth && truth.map((t, i) => <rect key={`t${i}`} {...rect(t)} className="det-truth" />)}
              {dets.map((d, i) => ({ d, i })).sort((a, b) => a.d.score - b.d.score).map(({ d, i }) => {
                const s = state.get(i)!;
                if (s === 'below') return null;
                const isCurrent = current?.kept === i;
                return (
                  <g key={i} className={`det-box det-${s}${isCurrent ? ' det-current' : ''}${selected === i ? ' det-selected' : ''}`}
                    onClick={() => setSelected(selected === i ? null : i)}>
                    <rect {...rect(d)} />
                    {(s === 'kept' || selected === i) && <text x={d.x0 + 2} y={d.y0 - 2}>{d.score.toFixed(2)}</text>}
                    {selected !== null && selected !== i && iou(dets[selected], d) > 0 && (
                      <text x={d.x1 - 2} y={d.y1 - 3} textAnchor="end" className="det-iou">IoU {iou(dets[selected], d).toFixed(2)}</text>
                    )}
                  </g>
                );
              })}
            </svg>
          </PixelCanvas>
          <p className="caption">Click a box to see its IoU with the boxes it overlaps.</p>
        </div>
        <div className="demo-panel det-side">
          <h3>NMS, step by step</h3>
          <div className="dl-stat-row">
            <span>Boxes ≥ score <strong>{dets.filter(d => d.score >= scoreThreshold).length}</strong></span>
            <span>Kept <strong>{keptCount}</strong></span>
            <span>Objects <strong>{truth.length}</strong></span>
          </div>
          <ol className="det-steps">
            {steps.slice(0, done).map((s, k) => (
              <li key={k} className={k === done - 1 ? 'det-steps-current' : undefined}>
                Keep <strong>{dets[s.kept].score.toFixed(2)}</strong>
                {s.suppressed.length > 0
                  ? <> → drop {s.suppressed.length} (IoU {s.suppressed.map(i => iou(dets[s.kept], dets[i]).toFixed(2)).join(', ')})</>
                  : ' → nothing overlaps enough'}
              </li>
            ))}
            {done < steps.length && <li className="muted">{steps.length - done} more step{steps.length - done > 1 ? 's' : ''}…</li>}
          </ol>
          <ul className="det-legend">
            <li><span className="det-key det-key-kept" /> kept</li>
            <li><span className="det-key det-key-suppressed" /> suppressed</li>
            <li><span className="det-key det-key-pending" /> not yet decided</li>
            {showTruth && <li><span className="det-key det-key-truth" /> ground truth</li>}
          </ul>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Why detectors need non-maximum suppression</h4>
        <p>A detector scores thousands of candidate boxes, and every object lights up many overlapping ones. NMS turns that
        cloud into one box per object: take the highest-scoring box, delete every remaining box that overlaps it by more than the
        <strong> IoU threshold</strong>, repeat.</p>
        <p className="formula"><code>IoU(A, B) = area(A ∩ B) / area(A ∪ B)</code></p>
        <p>Lower the IoU threshold and NMS becomes aggressive: two genuinely different objects that sit close together can merge
        into one. Raise it and duplicates survive. The <strong>score threshold</strong> throws away weak boxes before NMS even
        starts, removing most spurious detections, at the risk of losing a faint real object.</p>
      </div>
    </>
  );
}

const rect = (b: Box) => ({ x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0 });

// ---------------------------------------------------------------- precision / recall

function PrView({ image, truth, dets }: { image: ImageData; truth: Box[]; dets: Detection[] }) {
  const [iouThreshold, setIouThreshold] = useState(0.5);
  const [matchIou, setMatchIou] = useState(0.5);
  const raw = useMemo(() => evaluate(dets, dets.map((_, i) => i), truth, matchIou), [dets, truth, matchIou]);
  const kept = useMemo(
    () => evaluate(dets, nonMaxSuppression(dets, iouThreshold).map(s => s.kept), truth, matchIou),
    [dets, truth, iouThreshold, matchIou],
  );

  return (
    <>
      <div className="demo-controls">
        <label className="dl-slider">NMS IoU threshold <strong>{iouThreshold.toFixed(2)}</strong>
          <input type="range" min={0.1} max={0.95} step={0.05} value={iouThreshold} onChange={e => setIouThreshold(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Match if IoU ≥ <strong>{matchIou.toFixed(2)}</strong>
          <input type="range" min={0.3} max={0.9} step={0.05} value={matchIou} onChange={e => setMatchIou(Number(e.target.value))} />
        </label>
      </div>
      <div className="det-layout">
        <div className="demo-panel">
          <h3>Precision–recall curve</h3>
          <PrCurve curves={[{ label: 'without NMS', color: '#fb923c', ev: raw }, { label: 'after NMS', color: '#34d399', ev: kept }]} />
          <div className="dl-stat-row">
            <span>AP without NMS <strong>{raw.averagePrecision.toFixed(3)}</strong></span>
            <span>AP after NMS <strong>{kept.averagePrecision.toFixed(3)}</strong></span>
          </div>
        </div>
        <div className="demo-panel det-side">
          <h3>Ranked detections after NMS</h3>
          <PixelCanvas image={image} size={240} pixelated={false} label="Detections after NMS: true positives green, false positives red">
            <svg className="det-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`}>
              {truth.map((t, i) => <rect key={`t${i}`} {...rect(t)} className="det-truth" />)}
              {kept.ranked.map(r => <rect key={r.index} {...rect(dets[r.index])} className={r.truePositive ? 'det-tp' : 'det-fp'} />)}
            </svg>
          </PixelCanvas>
          <ol className="det-ranked">
            {kept.ranked.map((r, k) => (
              <li key={r.index} className={r.truePositive ? 'text-success' : 'text-error'}>
                {dets[r.index].score.toFixed(2)} {r.truePositive ? 'TP' : 'FP'} · P {kept.curve[k].precision.toFixed(2)} · R {kept.curve[k].recall.toFixed(2)}
              </li>
            ))}
          </ol>
          <p className="caption">{kept.truePositives} found · {kept.falsePositives} false alarms · {kept.missed} missed</p>
        </div>
      </div>
      <div className="demo-explanation">
        <h4>Average precision (AP)</h4>
        <p>Sort the detections by score and walk down the list. After each one, <strong>precision</strong> is the fraction of
        detections so far that are correct and <strong>recall</strong> the fraction of objects found. A detection only counts
        as correct if it overlaps an object not already claimed by a higher-scoring box with IoU ≥ the match threshold, so
        duplicates are false positives.</p>
        <p>AP is the area under this curve (with precision made monotonic). It rewards a detector that ranks its correct boxes
        above its mistakes. Without NMS the duplicates pile up as false positives and AP drops. Raising the match threshold
        (COCO averages it from 0.5 to 0.95) demands tighter boxes.</p>
      </div>
    </>
  );
}

function PrCurve({ curves }: { curves: { label: string; color: string; ev: Evaluation }[] }) {
  const W = 300, P = 30;
  const x = (r: number) => P + r * (W - 2 * P);
  const y = (p: number) => W - P - p * (W - 2 * P);
  return (
    <figure className="line-chart" style={{ maxWidth: 380 }}>
      <svg viewBox={`0 0 ${W} ${W}`} role="img" aria-label="Precision against recall">
        {[0, 0.25, 0.5, 0.75, 1].map(v => (
          <g key={v}>
            <line x1={x(0)} x2={x(1)} y1={y(v)} y2={y(v)} className="line-chart-grid" />
            <text x={x(0) - 4} y={y(v)} className="line-chart-tick" textAnchor="end" dominantBaseline="middle">{v}</text>
            <text x={x(v)} y={W - P + 12} className="line-chart-tick" textAnchor="middle">{v}</text>
          </g>
        ))}
        <text x={W / 2} y={W - 4} className="line-chart-tick" textAnchor="middle">recall →</text>
        <text x={8} y={P - 10} className="line-chart-tick">precision</text>
        {curves.map(c => (
          <path key={c.label} fill="none" stroke={c.color} strokeWidth={2}
            d={`M${x(0)},${y(1)}` + c.ev.curve.map(pt => `L${x(pt.recall)},${y(pt.precision)}`).join('')} />
        ))}
      </svg>
      <figcaption className="line-chart-legend">
        {curves.map(c => <span key={c.label} className="line-chart-key"><span className="line-chart-swatch" style={{ borderColor: c.color }} />{c.label}</span>)}
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- anchors

const SCALES = [24, 48, 96];
const RATIOS = [0.5, 1, 2];

function AnchorView({ image, truth }: { image: ImageData; truth: Box[] }) {
  const [grid, setGrid] = useState(8);
  const [scales, setScales] = useState([24, 48, 96]);
  const [ratios, setRatios] = useState([0.5, 1, 2]);
  const [positive, setPositive] = useState(0.5);
  const list = useMemo(() => anchors(SIZE, grid, scales, ratios), [grid, scales, ratios]);
  const labels = useMemo(() => labelAnchors(list, truth, positive, Math.min(0.4, positive - 0.05)), [list, truth, positive]);
  const count = (k: string) => labels.filter(l => l.kind === k).length;
  const pos = count('positive');
  const toggle = (arr: number[], v: number) => (arr.includes(v) ? (arr.length > 1 ? arr.filter(a => a !== v) : arr) : [...arr, v].sort((a, b) => a - b));

  return (
    <>
      <div className="demo-controls">
        <label className="dl-slider">Grid <strong>{grid}×{grid}</strong>
          <input type="range" min={2} max={16} value={grid} onChange={e => setGrid(Number(e.target.value))} />
        </label>
        <span className="demo-label">Sizes:</span>
        {SCALES.map(s => (
          <button key={s} type="button" aria-pressed={scales.includes(s)} className={`btn btn-sm${scales.includes(s) ? ' btn-primary' : ''}`}
            onClick={() => setScales(toggle(scales, s))}>{s}px</button>
        ))}
        <span className="demo-label">Aspect ratios:</span>
        {RATIOS.map(r => (
          <button key={r} type="button" aria-pressed={ratios.includes(r)} className={`btn btn-sm${ratios.includes(r) ? ' btn-primary' : ''}`}
            onClick={() => setRatios(toggle(ratios, r))}>{r === 1 ? '1:1' : r < 1 ? '1:2' : '2:1'}</button>
        ))}
        <label className="dl-slider">Positive if IoU ≥ <strong>{positive.toFixed(2)}</strong>
          <input type="range" min={0.3} max={0.8} step={0.05} value={positive} onChange={e => setPositive(Number(e.target.value))} />
        </label>
      </div>
      <div className="det-layout">
        <div className="demo-panel">
          <PixelCanvas image={image} size={DISPLAY} pixelated={false} label="Scene with the anchors assigned to each object">
            <svg className="det-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`}>
              {Array.from({ length: grid - 1 }, (_, i) => (i + 1) * (SIZE / grid)).map(v => (
                <g key={v} className="det-grid"><line x1={v} x2={v} y1={0} y2={SIZE} /><line x1={0} x2={SIZE} y1={v} y2={v} /></g>
              ))}
              {list.map((a, i) => labels[i].kind === 'ignored' ? <rect key={i} {...rect(a)} className="det-anchor-ignored" /> : null)}
              {list.map((a, i) => labels[i].kind === 'positive' ? <rect key={i} {...rect(a)} className="det-anchor-positive" /> : null)}
              {truth.map((t, i) => <rect key={`t${i}`} {...rect(t)} className="det-truth" />)}
            </svg>
          </PixelCanvas>
        </div>
        <div className="demo-panel det-side">
          <h3>Training labels</h3>
          <div className="det-counts">
            <div><strong>{list.length}</strong><span>anchors</span></div>
            <div className="text-success"><strong>{pos}</strong><span>positive</span></div>
            <div><strong>{count('ignored')}</strong><span>ignored</span></div>
            <div className="muted"><strong>{count('negative')}</strong><span>background</span></div>
          </div>
          <p className="caption">{pos > 0 ? `${Math.round(count('negative') / pos)} background anchors for every positive one.` : ''}</p>
          <ul className="det-legend">
            <li><span className="det-key det-key-kept" /> positive anchor</li>
            <li><span className="det-key det-key-pending" /> ignored (in between)</li>
            <li><span className="det-key det-key-truth" /> object</li>
          </ul>
        </div>
      </div>
      <div className="demo-explanation">
        <h4>Anchors: the boxes a detector starts from</h4>
        <p>Detectors such as Faster R-CNN, SSD and RetinaNet tile the image with <strong>anchor boxes</strong>: a few sizes and
        aspect ratios centred on every cell of a coarse grid (one grid per feature map). For each anchor the network predicts
        "is there an object?" and a small shift and stretch to fit it.</p>
        <p>For training, every anchor gets a label from its best IoU with the objects: positive above the threshold, background
        below 0.4, and ignored in between; each object also claims its best anchor so none is missed. Notice the imbalance: far
        more background than positive anchors. That is why RetinaNet introduced the <em>focal loss</em>, which down-weights the
        easy background examples. Remove the sizes or shapes that match your objects and see how few positives are left.</p>
      </div>
    </>
  );
}
