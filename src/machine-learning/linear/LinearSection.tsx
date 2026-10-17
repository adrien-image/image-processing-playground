import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import linearCode from './linear.ts?raw';
import pythonCode from './linear.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { cssColor, useTheme } from '../../lib/theme';
import { classMap, CLASS_COLORS, type Extent } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import {
  cluster, fitLogistic, leastSquares, lineCurvature, lineMse, lineStep, logLoss, mse, noisyLine, normalAngle,
  predictLinear, predictProba, sigmoid, thresholdAccuracy, twoBlobs, type LinearModel,
} from './linear';
import { clipLine, contourSegments, mseGrid } from './surface';
import '../shared/ml-shared.css';
import './linear.css';

type Mode = 'regression' | 'classification';

export function LinearSection() {
  const [mode, setMode] = useState<Mode>('regression');
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Task:</span>
        {(['regression', 'classification'] as const).map(m => (
          <button key={m} type="button" aria-pressed={mode === m} className={`btn btn-sm${mode === m ? ' btn-primary' : ''}`}
            onClick={() => setMode(m)}>{m === 'regression' ? 'Regression (least squares)' : 'Classification (logistic)'}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={linearCode} title="linear/linear.ts" python={{ code: pythonCode, title: 'linear/linear.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {mode === 'regression' ? <RegressionDemo /> : <ClassificationDemo />}
    </div>
  );
}

// ---- Regression ----

const DATA_EXTENT: Extent = { x: [-0.5, 2.5], y: [-1, 3] };
const BASE_VIEW: Extent = { x: [-1.5, 2.5], y: [-1.5, 2.5] };
const START: [number, number] = [-1, 2];
const RES = 120;
const PLOT = 360;
const MAX_STEPS = 3000;

interface Points { x: number[]; y: number[] }
interface Descent { path: [number, number][]; diverged: boolean }

const initialPoints = (): Points => noisyLine(30, seededRandom(4));
const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toFixed(digits)) : '∞');

function RegressionDemo() {
  const [points, setPoints] = useState<Points>(initialPoints);
  const [lr, setLr] = useState(0.3);
  const [descent, setDescent] = useState<Descent>({ path: [START], diverged: false });
  const [playing, setPlaying] = useState(false);

  const exact = useMemo(() => leastSquares(points.x, points.y, 1), [points]);
  const ew = exact.w[0], eb = exact.b;
  const limit = useMemo(() => 2 / lineCurvature(points.x)[0], [points]);
  const [gw, gb] = descent.path[descent.path.length - 1];
  const exactMse = lineMse(points.x, points.y, ew, eb);
  const gdMse = lineMse(points.x, points.y, gw, gb);
  const losses = useMemo(() => descent.path.map(([w, b]) => lineMse(points.x, points.y, w, b)), [descent, points]);

  // The loss view: the default box, grown if the minimum moves out of it.
  const view = useMemo<Extent>(() => ({
    x: [Math.min(BASE_VIEW.x[0], ew - 0.6), Math.max(BASE_VIEW.x[1], ew + 0.6)],
    y: [Math.min(BASE_VIEW.y[0], eb - 0.6), Math.max(BASE_VIEW.y[1], eb + 0.6)],
  }), [ew, eb]);
  const grid = useMemo(() => mseGrid(points.x, points.y, view, RES), [points, view]);

  /** Advance gradient descent by k steps (a pure function of the current state). */
  const advance = useCallback((d: Descent, k: number): Descent => {
    if (d.diverged) return d;
    const path = [...d.path];
    let [w, b] = path[path.length - 1];
    for (let s = 0; s < k && path.length <= MAX_STEPS; s++) {
      [w, b] = lineStep(points.x, points.y, w, b, lr);
      if (!Number.isFinite(w) || !Number.isFinite(b) || Math.abs(w) + Math.abs(b) > 1e6) return { path, diverged: true };
      path.push([w, b]);
    }
    return { path, diverged: false };
  }, [points, lr]);

  const done = descent.diverged || descent.path.length > MAX_STEPS || Math.hypot(gw - ew, gb - eb) < 1e-5;

  useEffect(() => {
    if (!playing) return;
    const id = requestAnimationFrame(() => {
      if (done) { setPlaying(false); return; }
      setDescent(advance(descent, descent.path.length < 60 ? 1 : 3));
    });
    return () => cancelAnimationFrame(id);
  }, [playing, done, descent, advance]);

  /** New data: gradient descent carries on from where it is, on the new loss. */
  const changePoints = (next: Points) => {
    setPoints(next);
    setDescent(d => ({ path: [d.diverged ? START : d.path[d.path.length - 1]], diverged: false }));
  };
  const addPoint = (x: number, y: number) => {
    const i = points.x.findIndex(v => v > x);
    const at = i < 0 ? points.x.length : i;
    changePoints({ x: points.x.toSpliced(at, 0, x), y: points.y.toSpliced(at, 0, y) });
  };
  const resetDescent = (start: [number, number] = START) => { setDescent({ path: [start], diverged: false }); };

  const layers = useMemo(() => [{ xy: Float32Array.from(points.x.flatMap((x, i) => [x, points.y[i]])), color: 'var(--series-blue)', size: 7 }], [points]);
  const overlay = useCallback(({ ctx, toPx, color }: Draw) => {
    // Residuals of the exact fit: the squared error is the sum of their squared lengths.
    ctx.strokeStyle = color('color-mix(in srgb, var(--series-green) 55%, transparent)');
    ctx.lineWidth = 1;
    ctx.beginPath();
    points.x.forEach((x, i) => {
      const [px, py] = toPx(x, points.y[i]), [, qy] = toPx(x, ew * x + eb);
      ctx.moveTo(px, py); ctx.lineTo(px, qy);
    });
    ctx.stroke();
    const line = (w: number, b: number, css: string, dash: number[]) => {
      const seg = clipLine(-w, 1, b, DATA_EXTENT); // y − w·x = b
      if (!seg) return;
      const [x0, y0] = toPx(seg[0], seg[1]), [x1, y1] = toPx(seg[2], seg[3]);
      ctx.strokeStyle = color(css); ctx.lineWidth = 2.5; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
    };
    line(ew, eb, 'var(--series-green)', []);
    if (Number.isFinite(gw) && Number.isFinite(gb)) line(gw, gb, 'var(--series-orange)', [7, 5]);
  }, [points, ew, eb, gw, gb]);

  const status = descent.diverged ? 'diverged' : done ? 'converged' : playing ? 'running' : 'paused';

  return (
    <>
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={() => { if (done) resetDescent(); setPlaying(p => !p || done); }}>
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button type="button" className="btn btn-sm" disabled={descent.diverged} onClick={() => { setPlaying(false); setDescent(advance(descent, 1)); }}>⏭ Step</button>
        <button type="button" className="btn btn-sm" onClick={() => { setPlaying(false); resetDescent(); }}>↺ Reset descent</button>
        <label className="dl-slider">Learning rate η <strong>{lr.toFixed(2)}</strong>
          <input type="range" min={0.01} max={0.6} step={0.01} value={lr} onChange={e => setLr(Number(e.target.value))} />
        </label>
        <span className="dl-status">stable below η = {limit.toFixed(2)}</span>
        <button type="button" className="btn btn-sm ml-auto" onClick={() => changePoints(initialPoints())}>↺ Reset points</button>
      </div>

      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Data · click to add a point</h3>
          <ScatterPlot layers={layers} extent={DATA_EXTENT} size={PLOT} overlay={overlay} onPick={addPoint}
            label="The data points with the exact least-squares line and the gradient-descent line" />
          <div className="ml-legend">
            <span><span className="lin-line" style={{ borderColor: 'var(--series-green)' }} />exact least squares</span>
            <span><span className="lin-line lin-dashed" style={{ borderColor: 'var(--series-orange)' }} />gradient descent</span>
          </div>
          <p className="caption">Thin green lines: the residuals of the exact fit. Squared, they add up to its error. Try a point far below the right end.</p>
        </div>

        <div className="demo-panel">
          <h3>Loss surface MSE(w, b) · click to start there</h3>
          <LossSurface grid={grid} path={descent.path} minimum={[ew, eb]} onPick={(w, b) => resetDescent([w, b])} />
          <div className="ml-legend">
            <span><span className="ml-swatch" style={{ background: 'var(--series-green)' }} />exact minimum</span>
            <span><span className="ml-swatch" style={{ background: 'var(--series-orange)' }} />gradient-descent path</span>
          </div>
          <p className="caption">Darker is lower; contours every equal step of log MSE. Slope w runs left to right, intercept b bottom to top.</p>
        </div>

        <div className="demo-panel">
          <h3>Error</h3>
          <dl className="ml-stats lin-stats">
            <dt>exact fit</dt><dd><strong>w {fmt(ew)} · b {fmt(eb)}</strong></dd>
            <dt>exact MSE</dt><dd><strong>{fmt(exactMse, 4)}</strong></dd>
            <dt>descent, step {descent.path.length - 1}</dt><dd><strong>w {fmt(gw)} · b {fmt(gb)}</strong></dd>
            <dt>descent MSE</dt><dd><strong>{descent.diverged ? 'diverged' : fmt(gdMse, 4)}</strong></dd>
            <dt>status</dt><dd><strong>{status}</strong></dd>
            <dt>points</dt><dd><strong>{points.x.length}</strong></dd>
          </dl>
          <LineChart title="Mean squared error of gradient descent against step, with the exact minimum" logY xLabel="step" height={200}
            yMax={Math.max(...losses) > 1e3 ? 1e3 : undefined}
            series={[
              { label: 'gradient descent', color: 'var(--series-orange)', values: losses },
              { label: 'exact minimum', color: 'var(--series-green)', values: losses.map(() => exactMse), dashed: true },
            ]} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>A straight line, two ways</h4>
        <p>The model is a line, ŷ = w·x + b, and least squares picks the w and b that minimise the mean squared error, the
        average squared vertical distance from each point to the line. That error is a quadratic bowl in (w, b) (the loss-surface
        plot), so its lowest point can be found exactly: set both partial derivatives to zero and solve the two
        <strong> normal equations</strong>. On the starting data this gives w = 0.618, b = 0.358 and an MSE of 0.051.</p>
        <p className="formula"><code>AᵀA θ = Aᵀy,  A = [x 1], θ = (w, b)    ·    θ ← θ − η · (2/n) Aᵀ(Aθ − y)</code></p>
        <p><strong>Gradient descent</strong> reaches the same point by walking downhill instead, which is how models without a
        closed form (logistic regression, every neural network) are trained. Because x runs from 0 to 2, w and b are tangled:
        raising the slope can be undone by lowering the intercept. So the bowl is a long, narrow valley, with curvatures 4.68
        across it and 0.26 along it. A step must be small enough for the steep direction, η &lt; 2 / 4.68 = 0.43, and then crawls
        along the flat one. At η = 0.3 it takes 97 steps to come within 0.001 of the minimum. Push η to 0.40 and the path
        zig-zags across the valley; past 0.43 every step overshoots further than the last and it <strong>diverges</strong>. This is
        the ravine of the <a href="#/optimizers">Optimizers</a> lesson, where momentum and Adam deal with it. Here, centring x
        (subtracting its mean) would make the bowl round.</p>
        <p><strong>Outliers.</strong> Squaring makes a residual of 2 cost 4 times a residual of 1, so one bad point can outweigh
        many good ones. Click once at (2.4, −0.9), below the right end: the slope falls from 0.62 to 0.29, and gradient descent
        (if it is running) slides to the new minimum. That sensitivity is the price of the closed form; robust losses such as
        the absolute or Huber error grow only linearly for large residuals.</p>
      </div>
    </>
  );
}

/** The loss surface as a heat map with contours, the descent path and the exact minimum. Redrawn on theme changes. */
function LossSurface({ grid, path, minimum, onPick }: {
  grid: ReturnType<typeof mseGrid>;
  path: [number, number][];
  minimum: [number, number];
  onPick: (w: number, b: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();
  const { view, res, values } = grid;

  // The contour lines depend only on the grid; colours are resolved while drawing.
  const background = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const levels = Array.from({ length: 12 }, (_, k) => lo + ((k + 0.5) / 12) * (hi - lo));
    return { lo, hi, levels: levels.map(l => contourSegments(values, res, l)) };
  }, [values, res]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = PLOT * dpr; canvas.height = PLOT * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const rgb = (css: string) => (cssColor(css).match(/[\d.]+/g) ?? ['128', '128', '128']).slice(0, 3).map(Number);
    const deep = rgb('var(--series-blue)'), paper = rgb('var(--surface-solid)');
    const img = new ImageData(res, res);
    const { lo, hi } = background;
    for (let i = 0; i < res * res; i++) {
      const t = Math.pow((values[i] - lo) / (hi - lo || 1), 0.9);
      img.data.set([0, 1, 2].map(c => deep[c] + (paper[c] - deep[c]) * (0.15 + 0.85 * t)).concat(255), i * 4);
    }
    const off = document.createElement('canvas');
    off.width = off.height = res;
    off.getContext('2d')?.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, PLOT, PLOT);

    // Contours, in grid units (cell centres) → pixels.
    const g = (v: number) => ((v + 0.5) / res) * PLOT;
    ctx.strokeStyle = cssColor('color-mix(in srgb, var(--fg) 28%, transparent)');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const segs of background.levels) for (let k = 0; k < segs.length; k += 4) {
      ctx.moveTo(g(segs[k]), g(segs[k + 1])); ctx.lineTo(g(segs[k + 2]), g(segs[k + 3]));
    }
    ctx.stroke();

    const toPx = (w: number, b: number): [number, number] => [
      ((w - view.x[0]) / (view.x[1] - view.x[0])) * PLOT, ((view.y[1] - b) / (view.y[1] - view.y[0])) * PLOT,
    ];
    const orange = cssColor('var(--series-orange)'), outline = cssColor('var(--surface-solid)');
    ctx.strokeStyle = orange; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath();
    path.forEach(([w, b], i) => { const [x, y] = toPx(w, b); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke();
    // Small dots on the first steps, where the zig-zag is visible.
    ctx.fillStyle = orange;
    for (const [w, b] of path.slice(0, 40)) { const [x, y] = toPx(w, b); ctx.beginPath(); ctx.arc(x, y, 2, 0, 2 * Math.PI); ctx.fill(); }
    const [sx, sy] = toPx(...path[0]);
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, 2 * Math.PI); ctx.stroke();
    const [hx, hy] = toPx(...path[path.length - 1]);
    ctx.beginPath(); ctx.arc(hx, hy, 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = outline; ctx.lineWidth = 1.5; ctx.stroke();
    const [mx, my] = toPx(...minimum);
    ctx.fillStyle = cssColor('var(--series-green)');
    ctx.beginPath(); ctx.arc(mx, my, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }, [background, values, res, view, path, minimum, theme]);

  const pick = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width, fy = (e.clientY - rect.top) / rect.height;
    onPick(view.x[0] + fx * (view.x[1] - view.x[0]), view.y[1] - fy * (view.y[1] - view.y[0]));
  };

  return (
    <div className="lin-surface">
      <canvas ref={ref} className="ml-plot ml-plot-pick" style={{ width: PLOT, maxWidth: '100%', aspectRatio: '1' }} onPointerDown={pick}
        role="img" aria-label="Heat map of the mean squared error over slope w and intercept b, with the gradient-descent path and the exact minimum" />
      <span className="lin-axis lin-axis-x">w: {view.x[0].toFixed(1)} … {view.x[1].toFixed(1)}</span>
      <span className="lin-axis lin-axis-y">b: {view.y[0].toFixed(1)} … {view.y[1].toFixed(1)}</span>
    </div>
  );
}

// ---- Classification ----

const CLASS_EXTENT: Extent = { x: [-3, 3], y: [-3, 3] };
const FAR: [number, number] = [1.4, 2.8];
const LOGISTIC = { lr: 1, steps: 1000, l2: 1e-3 };
/** The loss of one point against its margin m (labels coded ±1): x from −2 to 4. */
const MARGINS = Array.from({ length: 61 }, (_, i) => -2 + i * 0.1);

interface Added { xy: number[]; labels: number[] }

function ClassificationDemo() {
  const [separation, setSeparation] = useState(1.2);
  const [spread, setSpread] = useState(0.25);
  const [added, setAdded] = useState<Added>({ xy: [], labels: [] });
  const [addClass, setAddClass] = useState<0 | 1>(1);

  const base = useMemo(() => twoBlobs(100, separation, spread, seededRandom(9)), [separation, spread]);
  const xy = useMemo(() => [...base.xy, ...added.xy], [base, added]);
  const labels = useMemo(() => [...base.labels, ...added.labels], [base, added]);

  const logistic = useMemo(() => fitLogistic(xy, labels, 2, LOGISTIC), [xy, labels]);
  const ls = useMemo(() => leastSquares(xy, labels, 2), [xy, labels]);
  // The fits without the added points, to measure how far each boundary turned.
  const logistic0 = useMemo(() => fitLogistic(base.xy, base.labels, 2, LOGISTIC), [base]);
  const ls0 = useMemo(() => leastSquares(base.xy, base.labels, 2), [base]);

  const map = useMemo(() => classMap((x, y) => {
    const p = sigmoid(logistic.w[0] * x + logistic.w[1] * y + logistic.b);
    return [1 - p, p];
  }, 2, 72, CLASS_EXTENT), [logistic]);

  const stats = useMemo(() => {
    const p = predictProba(logistic, xy, 2), f = predictLinear(ls, xy, 2);
    const pBase = predictProba(logistic, base.xy, 2), fBase = predictLinear(ls, base.xy, 2);
    return {
      logLoss: logLoss(p, labels), mse: mse(ls, xy, labels, 2),
      accLogistic: thresholdAccuracy(p, labels, 0.5), accLs: thresholdAccuracy(f, labels, 0.5),
      baseLogistic: thresholdAccuracy(pBase, base.labels, 0.5), baseLs: thresholdAccuracy(fBase, base.labels, 0.5),
      turnLogistic: angleBetween(logistic, logistic0), turnLs: angleBetween(ls, ls0),
    };
  }, [logistic, ls, logistic0, ls0, xy, labels, base]);

  const layers = useMemo(() => [
    { xy: Float32Array.from(base.xy), labels: base.labels, size: 7 },
    { xy: Float32Array.from(added.xy), labels: added.labels, size: 7 },
    { xy: Float32Array.from(added.xy), labels: added.labels, size: 6, shape: 'ring' as const },
  ], [base, added]);

  const overlay = useCallback(({ ctx, toPx, color }: Draw) => {
    const line = (m: LinearModel, level: number, css: string, dash: number[]) => {
      const seg = clipLine(m.w[0], m.w[1], level - m.b, CLASS_EXTENT);
      if (!seg) return;
      const [x0, y0] = toPx(seg[0], seg[1]), [x1, y1] = toPx(seg[2], seg[3]);
      ctx.strokeStyle = color(css); ctx.lineWidth = 2.5; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
    };
    line(ls, 0.5, 'var(--series-red)', [8, 6]);
    line(logistic, 0, 'var(--text)', []);
  }, [logistic, ls]);

  const addPoints = (pts: number[], label: number) =>
    setAdded(a => ({ xy: [...a.xy, ...pts], labels: [...a.labels, ...pts.filter((_, i) => i % 2 === 0).map(() => label)] }));

  const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
  const marginLosses = useMemo(() => ({
    squared: MARGINS.map(m => (1 - m) ** 2),
    logistic: MARGINS.map(m => Math.log1p(Math.exp(-m)) / Math.LN2),
    zeroOne: MARGINS.map(m => (m < 0 ? 1 : 0)),
  }), []);

  return (
    <>
      <div className="demo-controls">
        <label className="dl-slider">Separation <strong>{separation.toFixed(1)}</strong>
          <input type="range" min={0.2} max={2.5} step={0.1} value={separation} onChange={e => setSeparation(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Spread <strong>{spread.toFixed(2)}</strong>
          <input type="range" min={0.1} max={0.6} step={0.05} value={spread} onChange={e => setSpread(Number(e.target.value))} />
        </label>
        <span className="demo-label">Click adds class:</span>
        {([0, 1] as const).map(c => (
          <button key={c} type="button" aria-pressed={addClass === c} className={`btn btn-sm${addClass === c ? ' btn-primary' : ''}`} onClick={() => setAddClass(c)}>
            <span className="ml-swatch" style={{ background: CLASS_COLORS[c] }} />class {c}
          </button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => addPoints(cluster(FAR[0], FAR[1], 10, seededRandom(5 + added.xy.length)), 1)}>
          + 10 far class-1 points
        </button>
        <button type="button" className="btn btn-sm" disabled={!added.xy.length} onClick={() => setAdded({ xy: [], labels: [] })}>↺ Remove added</button>
      </div>

      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Two classes · click to add a point</h3>
          <ScatterPlot layers={layers} map={map} extent={CLASS_EXTENT} size={PLOT} overlay={overlay} onPick={(x, y) => addPoints([x, y], addClass)}
            label="Two classes of points over the logistic regression's probability map, with the logistic and least-squares boundaries" />
          <div className="ml-legend">
            <span><span className="lin-line" style={{ borderColor: 'var(--text)' }} />logistic, P = 0.5</span>
            <span><span className="lin-line lin-dashed" style={{ borderColor: 'var(--series-red)' }} />least squares on 0/1, f = 0.5</span>
            <span><span className="lin-ring" />added</span>
          </div>
          <p className="caption">Background: the logistic model's P(class 1), from blue (class 0) to orange (class 1), paler where it is unsure.</p>
        </div>

        <div className="demo-panel">
          <h3>Two fits of the same points</h3>
          <table className="lin-table">
            <thead><tr><th scope="col">measure</th><th scope="col">logistic</th><th scope="col">least squares</th></tr></thead>
            <tbody>
              <tr><th scope="row">training loss <span className="muted">(log-loss · MSE)</span></th><td>{stats.logLoss.toFixed(3)}</td><td>{stats.mse.toFixed(3)}</td></tr>
              <tr><th scope="row">accuracy, all {labels.length} points</th><td>{pct(stats.accLogistic)}</td><td>{pct(stats.accLs)}</td></tr>
              <tr><th scope="row">accuracy, the 100 original points</th><td>{pct(stats.baseLogistic)}</td><td>{pct(stats.baseLs)}</td></tr>
              <tr><th scope="row">boundary turned by the added points</th><td>{stats.turnLogistic.toFixed(1)}°</td><td>{stats.turnLs.toFixed(1)}°</td></tr>
            </tbody>
          </table>
          <LineChart title="Logistic regression's training log-loss over gradient-descent steps" xLabel="gradient-descent step (η = 1)" height={170}
            series={[{ label: 'log-loss', color: 'var(--text)', values: logistic.losses.filter((_, i) => i % 10 === 0) }]} xTickLabel={i => String(i * 10)} />
          <p className="caption">Logistic regression has no closed form: {LOGISTIC.steps} full-batch gradient-descent steps from w = 0, refitted at every change.</p>
        </div>

        <div className="demo-panel">
          <h3>What one point costs</h3>
          <LineChart title="Squared, logistic and 0/1 loss of one point against its margin" xLabel="margin m (how far on the correct side)" height={220}
            xTickLabel={i => MARGINS[i].toFixed(0)}
            series={[
              { label: 'squared error (1 − m)²', color: 'var(--series-red)', values: marginLosses.squared, dashed: true },
              { label: 'logistic log₂(1 + e⁻ᵐ)', color: 'var(--text)', values: marginLosses.logistic },
              { label: '0/1 error', color: 'var(--text-muted)', values: marginLosses.zeroOne },
            ]} />
          <p className="caption">With labels coded ±1, m = label × score. Past m = 1 the squared error rises again: least squares
          punishes a point for being too right.</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Why classification needs its own loss</h4>
        <p>Nothing stops us fitting the labels 0 and 1 with least squares and calling f(x) &gt; 0.5 class 1: that is the dashed
        line, and on two tidy blobs it is a reasonable boundary. <strong>Logistic regression</strong> keeps the same linear score
        but squashes it into a probability, p = σ(w·x + b), and maximises the likelihood of the labels, which means minimising
        the cross-entropy. Its gradient has the same shape as least squares', with the probability in place of the raw score:</p>
        <p className="formula"><code>L = −mean[ y log p + (1 − y) log(1 − p) ],   ∇L = (1/n) Aᵀ(p − y)</code></p>
        <p>The difference is in what a <strong>correct, confident</strong> point contributes. Press "+ 10 far class-1 points":
        they sit far on the right side of both boundaries, so they should change nothing. For logistic regression they are
        already at p ≈ 1, their gradient p − y is about zero, and the solid line does not move (0.0° here). For least squares
        their score is far above 1, so their squared error is large, and the fit tilts to bring them down: the dashed line turns
        by about 24° and starts misclassifying the original points (100% → 95% on the default blobs). The chart on the right
        shows why: the squared error is a parabola with its minimum at margin 1, while the logistic loss keeps falling.</p>
        <p>Move the blobs apart until they no longer overlap: the logistic weights would grow without bound (the data can be
        separated perfectly, so ever-steeper sigmoids keep lowering the loss), which is why the fit carries a small L2 penalty
        (λ = 0.001). Both models draw a straight boundary; for curved ones see k-nearest neighbours, SVM kernels or trees.</p>
      </div>
    </>
  );
}

/** The angle in degrees between two boundaries' normals. */
function angleBetween(a: LinearModel, b: LinearModel): number {
  const d = Math.abs(normalAngle(a) - normalAngle(b)) % 360;
  return Math.min(d, 360 - d);
}
