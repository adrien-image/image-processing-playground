import { useCallback, useMemo, useState } from 'react';
import svmCode from './svm.ts?raw';
import pythonCode from './svm.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { seededRandom } from '../../lib/random';
import { makeClassification, SHAPE_LABELS, type Shape2D } from '../shared/datasets';
import { CLASS_COLORS, UNIT_EXTENT, type ClassMap } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import { contourSegments, joinSegments } from './contour';
import { decision, marginWidth, objective, primalWeights, supportVectors, trainSvm, type Kernel } from './svm';
import '../shared/ml-shared.css';
import './svm.css';

const SHAPES: Shape2D[] = ['blobs', 'moons', 'circles', 'xor'];
const N = 120;
const MAX_POINTS = 300;
const RES = 90;
const extent = UNIT_EXTENT;

interface Points { xy: number[]; labels: number[] }
type ClickMode = 0 | 1 | 'remove';
type KernelType = Kernel['type'];
const KERNEL_LABELS: Record<KernelType, string> = { linear: 'Linear', rbf: 'RBF (Gaussian)', poly: 'Polynomial' };

/** Fresh points from the same distribution, to measure how well the boundary generalises. */
const testSet = (shape: Shape2D, noise: number) => makeClassification(shape, 1000, noise, seededRandom(99));

const generate = (shape: Shape2D, noise: number): Points => {
  const d = makeClassification(shape, N, noise, seededRandom(1));
  return { xy: Array.from(d.xy), labels: Array.from(d.labels) };
};

/** f on the plot's grid of cell centres (row 0 at the top), as classMap lays it out. */
function decisionGrid(f: (x: number, y: number) => number): Float64Array {
  const values = new Float64Array(RES * RES);
  for (let row = 0; row < RES; row++) for (let col = 0; col < RES; col++) {
    const x = extent.x[0] + ((col + 0.5) / RES) * (extent.x[1] - extent.x[0]);
    const y = extent.y[1] - ((row + 0.5) / RES) * (extent.y[1] - extent.y[0]);
    values[row * RES + col] = f(x, y);
  }
  return values;
}

export function SvmSection() {
  const [shape, setShape] = useState<Shape2D>('blobs');
  const [noise, setNoise] = useState(0.1);
  const [points, setPoints] = useState<Points>(() => generate('blobs', 0.1));
  const [kernelType, setKernelType] = useState<KernelType>('linear');
  const [logC, setLogC] = useState(0);
  const [logGamma, setLogGamma] = useState(0.5);
  const [degree, setDegree] = useState(3);
  const [mode, setMode] = useState<ClickMode>(0);
  const [removal, setRemoval] = useState<{ removed: number; before: Float64Array } | null>(null);
  const [showCode, setShowCode] = useState(false);

  const C = 10 ** logC, gamma = 10 ** logGamma;
  const kernel = useMemo<Kernel>(() => (kernelType === 'rbf' ? { type: 'rbf', gamma } : kernelType === 'poly' ? { type: 'poly', degree } : { type: 'linear' }),
    [kernelType, gamma, degree]);

  const changeData = (next: Points) => { setPoints(next); setRemoval(null); };
  const chooseShape = (s: Shape2D) => { setShape(s); changeData(generate(s, noise)); };

  const test = useMemo(() => testSet(shape, noise), [shape, noise]);
  const bothClasses = points.labels.includes(0) && points.labels.includes(1);
  const result = useMemo(() => {
    if (!bothClasses) return null;
    const model = trainSvm(points.xy, points.labels.map(l => (l ? 1 : -1)), 2, kernel, C);
    const grid = decisionGrid((x, y) => decision(model, [x, y]));
    const lines = (level: number) => joinSegments(contourSegments(grid, RES, level, extent));
    // Background: the sign of f picks the class colour, |f| how strongly (saturating beyond the margin).
    const probs = new Float32Array(RES * RES * 2);
    grid.forEach((f, i) => { const p = 1 / (1 + Math.exp(-2.5 * f)); probs[2 * i] = 1 - p; probs[2 * i + 1] = p; });
    const map: ClassMap = { res: RES, classes: 2, probs };
    const { support, bounded } = supportVectors(model);
    const margins = points.labels.map((l, i) => (l ? 1 : -1) * decision(model, [points.xy[2 * i], points.xy[2 * i + 1]]));
    const contours = { zero: lines(0), plus: lines(1), minus: lines(-1) };
    let right = 0;
    for (let i = 0; i < test.n; i++) if ((decision(model, [test.xy[2 * i], test.xy[2 * i + 1]]) > 0 ? 1 : 0) === test.labels[i]) right++;
    return { model, grid, map, support, bounded, margins, contours, objective: objective(model), margin: marginWidth(model), testAccuracy: right / test.n };
  }, [points, kernel, C, bothClasses, test]);

  const maxChange = useMemo(() => {
    if (!removal || !result) return null;
    let m = 0;
    result.grid.forEach((v, i) => { m = Math.max(m, Math.abs(v - removal.before[i])); });
    return m;
  }, [removal, result]);

  const overlay = useCallback(({ ctx, toPx, color }: Draw) => {
    if (!result) return;
    const stroke = (lines: number[][], dashed: boolean) => {
      ctx.setLineDash(dashed ? [6, 5] : []);
      for (const line of lines) {
        ctx.beginPath();
        for (let k = 0; k < line.length; k += 2) ctx[k ? 'lineTo' : 'moveTo'](...toPx(line[k], line[k + 1]));
        ctx.stroke();
      }
    };
    ctx.strokeStyle = color('var(--text)');
    ctx.lineWidth = 2.2;
    stroke(result.contours.zero, false);
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.75;
    stroke(result.contours.plus, true);
    stroke(result.contours.minus, true);
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }, [result]);

  const layers = useMemo(() => {
    const xy = Float32Array.from(points.xy);
    return [
      { xy, labels: points.labels, size: 7 },
      ...(result ? [{ xy, color: 'var(--text)', shape: 'ring' as const, size: 7, mask: result.support }] : []),
    ];
  }, [points, result]);

  const pick = (x: number, y: number) => {
    if (mode === 'remove') {
      let best = -1, bestD = 0.1;
      for (let i = 0; i < points.labels.length; i++) {
        const d = Math.hypot(points.xy[2 * i] - x, points.xy[2 * i + 1] - y);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best < 0) return;
      changeData({ xy: points.xy.filter((_, k) => k >> 1 !== best), labels: points.labels.filter((_, k) => k !== best) });
    } else if (points.labels.length < MAX_POINTS) {
      changeData({ xy: [...points.xy, x, y], labels: [...points.labels, mode] });
    }
  };

  const removeOthers = () => {
    if (!result) return;
    const keep = result.support.map((s, i) => (s ? i : -1)).filter(i => i >= 0);
    setPoints({ xy: keep.flatMap(i => [points.xy[2 * i], points.xy[2 * i + 1]]), labels: keep.map(i => points.labels[i]) });
    setRemoval({ removed: points.labels.length - keep.length, before: result.grid });
  };

  const svCount = result ? result.support.filter(Boolean).length : 0;
  const atC = result ? result.bounded.filter(Boolean).length : 0;
  const w = result && kernelType === 'linear' ? primalWeights(result.model) : null;
  const accuracy = result ? result.margins.filter(m => m > 0).length / result.margins.length : 0;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {SHAPES.map(s => (
          <button key={s} type="button" aria-pressed={shape === s} className={`btn btn-sm${shape === s ? ' btn-primary' : ''}`}
            onClick={() => chooseShape(s)}>{SHAPE_LABELS[s]}</button>
        ))}
        <label className="dl-slider">Noise <strong>{noise.toFixed(2)}</strong>
          <input type="range" min={0} max={0.4} step={0.01} value={noise}
            onChange={e => { const v = Number(e.target.value); setNoise(v); changeData(generate(shape, v)); }} />
        </label>
        <button type="button" className="btn btn-sm" onClick={() => changeData(generate(shape, noise))}>↺ Reset</button>
        <span className="ml-auto">
          <ShowCodeButton code={svmCode} title="svm/svm.ts" python={{ code: pythonCode, title: 'svm/svm.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <span className="demo-label">Kernel:</span>
        {(Object.keys(KERNEL_LABELS) as KernelType[]).map(k => (
          <button key={k} type="button" aria-pressed={kernelType === k} className={`btn btn-sm${kernelType === k ? ' btn-primary' : ''}`}
            onClick={() => setKernelType(k)}>{KERNEL_LABELS[k]}</button>
        ))}
        <div className="dl-sliders">
          <label className="dl-slider">C <strong>{C < 1 ? C.toPrecision(2) : Math.round(C)}</strong>
            <input type="range" min={-2} max={3} step={0.05} value={logC} onChange={e => setLogC(Number(e.target.value))} />
          </label>
          {kernelType === 'rbf' && (
            <label className="dl-slider">γ <strong>{gamma < 10 ? gamma.toPrecision(2) : Math.round(gamma)}</strong>
              <input type="range" min={-1} max={2.5} step={0.05} value={logGamma} onChange={e => setLogGamma(Number(e.target.value))} />
            </label>
          )}
          {kernelType === 'poly' && (
            <label className="dl-slider">Degree <strong>{degree}</strong>
              <input type="range" min={2} max={6} step={1} value={degree} onChange={e => setDegree(Number(e.target.value))} />
            </label>
          )}
        </div>
      </div>

      <div className="demo-controls">
        <span className="demo-label">Click the plot to:</span>
        {([0, 1] as const).map(c => (
          <button key={c} type="button" aria-pressed={mode === c} className={`btn btn-sm${mode === c ? ' btn-primary' : ''}`} onClick={() => setMode(c)}>
            <span className="ml-swatch" style={{ background: CLASS_COLORS[c] }} />add {c ? '+1' : '−1'}
          </button>
        ))}
        <button type="button" aria-pressed={mode === 'remove'} className={`btn btn-sm${mode === 'remove' ? ' btn-primary' : ''}`}
          onClick={() => setMode('remove')}>✕ remove a point</button>
        <button type="button" className="btn btn-sm" onClick={removeOthers} disabled={!result || svCount === points.labels.length}>
          Keep only the support vectors
        </button>
      </div>

      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Decision function f(x)</h3>
          <ScatterPlot layers={layers} map={result?.map} overlay={overlay} onPick={pick} size={400}
            label="Training points over the SVM's decision function, with the boundary, the margins and the support vectors ringed" />
          <div className="ml-legend">
            <span><span className="ml-swatch" style={{ background: CLASS_COLORS[0] }} />y = −1</span>
            <span><span className="ml-swatch" style={{ background: CLASS_COLORS[1] }} />y = +1</span>
            <span><span className="svm-key-line" />f = 0</span>
            <span><span className="svm-key-line svm-key-dashed" />f = ±1 (margins)</span>
            <span><span className="svm-key-ring" />support vector</span>
          </div>
          {!bothClasses && <p className="error-banner">Both classes need at least one point.</p>}
          {removal && maxChange !== null && (
            <p className="caption svm-removal">Removed {removal.removed} points that were not support vectors: f changed by at most {maxChange.toFixed(4)} anywhere on the plot (SMO stops at a tolerance of 0.001).</p>
          )}
        </div>

        <div className="demo-panel">
          <h3>Who holds the boundary up</h3>
          {result && <KktPlot margins={result.margins} alpha={Array.from(result.model.alpha)} C={C} labels={points.labels} />}
          <p className="caption">Each training point at its margin y·f(x) (across) and its dual weight α (up). Points beyond the
          margin (y·f &gt; 1) have α = 0; points exactly on it have 0 &lt; α &lt; C; points inside it or misclassified are held at α = C.</p>
          {result && (
            <div className="ml-stats">
              <span>support vectors</span><strong>{svCount} of {points.labels.length}{atC ? ` (${atC} at α = C)` : ''}</strong>
              <span>margin width 2/‖w‖{kernelType === 'linear' ? '' : ' (feature space)'}</span><strong>{result.margin.toFixed(3)}</strong>
              {w && <><span>w, b</span><strong>({w[0].toFixed(2)}, {w[1].toFixed(2)}), {result.model.b.toFixed(2)}</strong></>}
              <span>training accuracy</span><strong>{(100 * accuracy).toFixed(1)}%</strong>
              <span>accuracy on 1000 fresh points</span><strong>{(100 * result.testAccuracy).toFixed(1)}%</strong>
              <span>inside the margin (y·f &lt; 1)</span><strong>{result.objective.violations}</strong>
              <span>½‖w‖² + C·Σ hinge</span><strong>{result.objective.value.toPrecision(4)}</strong>
              <span>SMO pair updates</span><strong>{result.model.iterations}{result.model.converged ? '' : ' (stopped early)'}</strong>
            </div>
          )}
        </div>
      </div>

      <div className="demo-explanation">
        <h4>The widest street between the classes</h4>
        <p>Many lines separate two clean blobs. The SVM picks the one with the widest empty band around it: the solid line is the
        boundary f(x) = 0 and the dashed lines are the margins f(x) = ±1. Scaling f so that the closest points sit exactly on
        the margins, the band is 2/‖w‖ wide, so the widest band is the smallest ‖w‖. Real data overlap, so the soft-margin SVM
        lets points into the band, or across it, and charges for it with the <strong>hinge loss</strong>:</p>
        <p className="formula"><code>minimise ½‖w‖² + C · Σᵢ max(0, 1 − yᵢ f(xᵢ)),   f(x) = w·x + b</code></p>
        <p>A point beyond its margin costs nothing; inside, it costs in proportion to how far. <strong>C</strong> sets the price:
        slide it down and the street widens and swallows more points; slide it up and the SVM bends over backwards to classify
        every training point, with a narrow margin that a single outlier can tilt. On the blobs with the linear kernel, C = 0.01
        gives a street 2.88 wide with all 120 points inside it, C = 1000 one 0.05 wide held by 3 points.</p>
        <p><strong>Only the support vectors matter.</strong> The solution is f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b, and αᵢ is exactly zero
        for every point beyond its margin: the right-hand plot shows it. Press <em>Keep only the support vectors</em>, or remove
        points far from the boundary one by one: the line does not move, and the caption measures it. Remove a ringed point
        and it does. SMO finds the αs by repeatedly taking the pair of points that most violates these conditions and solving
        for that pair exactly.</p>
        <p><strong>The kernel trick.</strong> The training points enter only through K(xᵢ, xⱼ), so any sensible similarity can
        stand in for the dot product. The RBF kernel exp(−γ‖x − z‖²) puts a bump on every support vector, and f adds them up:
        it separates the circles and XOR, where a line cannot. γ is the inverse squared width of the bumps. Push it to the top
        and each point gets its own little island: on Two moons with noise 0.2 and C = 1, γ = 3.2 keeps 45 of the 120 points as
        support vectors and gets 92.7% of fresh points right; γ = 316 keeps 119, classifies every training point correctly and
        only 76.1% of fresh ones. That is memorisation, and C and γ together are what you tune to avoid it.
        The polynomial kernel (x·z + 1)ᵈ is a linear SVM on every product of up to d coordinates.</p>
      </div>
    </div>
  );
}

/** α/C against y·f(x) for every training point: the KKT conditions, made visible. */
function KktPlot({ margins, alpha, C, labels }: { margins: number[]; alpha: number[]; C: number; labels: number[] }) {
  const W = 360, H = 230, pad = { l: 34, r: 10, t: 22, b: 30 };
  const lo = -1.5, hi = 3;
  const px = (m: number) => pad.l + ((Math.min(hi, Math.max(lo, m)) - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const py = (a: number) => H - pad.b - a * (H - pad.t - pad.b);
  return (
    <svg className="svm-kkt" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Dual weight alpha of each training point against its margin y times f(x)">
      <rect x={px(1)} y={pad.t} width={px(hi) - px(1)} height={H - pad.t - pad.b} className="svm-kkt-safe" />
      {[0, 0.5, 1].map(a => <g key={a}><line x1={pad.l} x2={W - pad.r} y1={py(a)} y2={py(a)} className="svm-kkt-grid" />
        <text x={pad.l - 4} y={py(a)} className="svm-kkt-tick" textAnchor="end" dominantBaseline="middle">{a === 1 ? 'C' : a ? 'C/2' : '0'}</text></g>)}
      {[-1, 0, 1, 2, 3].map(m => <text key={m} x={px(m)} y={H - pad.b + 13} className="svm-kkt-tick" textAnchor="middle">{m}</text>)}
      <line x1={px(1)} x2={px(1)} y1={pad.t} y2={H - pad.b} className="svm-kkt-one" />
      <line x1={px(0)} x2={px(0)} y1={pad.t} y2={H - pad.b} className="svm-kkt-grid" />
      <text x={W - pad.r} y={H - 3} className="svm-kkt-tick" textAnchor="end">y·f(x) →</text>
      <text x={px(1)} y={pad.t - 8} className="svm-kkt-tick" textAnchor="middle">y·f = 1</text>
      {margins.map((m, i) => (
        <circle key={i} cx={px(m)} cy={py(alpha[i] / C)} r={3.5} className="svm-kkt-dot" style={{ fill: CLASS_COLORS[labels[i]] }} />
      ))}
    </svg>
  );
}
