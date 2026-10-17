import { useDeferredValue, useMemo, useState } from 'react';
import selectionCode from './selection.ts?raw';
import pythonCode from './selection.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { makeRegression, trueCurve } from '../shared/datasets';
import { CLASS_COLORS, type Extent } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import { biasVariance, crossValidate, errorCurves, evaluate, fitPolynomial, kFolds, MAX_DEGREE, type ErrorCurves } from './selection';
import '../shared/ml-shared.css';
import './model-selection.css';

const EXTENT: Extent = { x: [-1.1, 1.1], y: [-1.7, 1.7] };
const DATASETS = 50;
const VALIDATION_SEED = 10007;
const FOLD_SEED = 31;

/** Ten fold colours: the six series colours, then four mixes of them with the text colour. */
const FOLD_COLORS = [
  ...CLASS_COLORS,
  ...['blue', 'orange', 'green', 'pink'].map(c => `color-mix(in srgb, var(--series-${c}) 55%, var(--text))`),
];

const fmt = (v: number) => (!Number.isFinite(v) ? '—' : v >= 100 ? v.toExponential(1) : v >= 0.1 ? v.toFixed(3) : v.toPrecision(2));

/** A curve y = f(x) across the plot. */
function drawCurve(d: Draw, f: (x: number) => number, colour: string, width: number, dash: number[] = []) {
  const { ctx } = d;
  ctx.beginPath();
  for (let i = 0; i <= 240; i++) {
    const x = EXTENT.x[0] + ((EXTENT.x[1] - EXTENT.x[0]) * i) / 240;
    const [px, py] = d.toPx(x, Math.max(-50, Math.min(50, f(x))));
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

const W = 480, H = 230, PAD = { top: 12, right: 12, bottom: 28, left: 46 };

/** Train, validation and CV error against degree on a log scale; click a degree to select it. */
function ErrorChart({ curves, degree, noise, onSelect }: { curves: ErrorCurves; degree: number; noise: number; onSelect: (d: number) => void }) {
  const all = [...curves.train, ...curves.validation, ...curves.cv].filter(v => Number.isFinite(v) && v > 0);
  const lo = Math.log10(Math.max(Math.min(...all), (noise * noise) / 30));
  const hi = Math.log10(Math.min(Math.max(...all), 30 * curves.validation[0]));
  const bottom = Math.floor(lo), top = Math.max(Math.ceil(hi), bottom + 1);
  const x = (d: number) => PAD.left + (d / MAX_DEGREE) * (W - PAD.left - PAD.right);
  const y = (v: number) => {
    const t = (Math.log10(Math.max(v, 1e-300)) - bottom) / (top - bottom);
    return PAD.top + (1 - Math.min(1, Math.max(0, t))) * (H - PAD.top - PAD.bottom);
  };
  const line = (values: number[]) => values.map((v, d) => `${d ? 'L' : 'M'}${x(d).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const ticks = Array.from({ length: top - bottom + 1 }, (_, i) => bottom + i);
  const step = (W - PAD.left - PAD.right) / MAX_DEGREE;
  return (
    <figure className="line-chart ms-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Mean squared error against polynomial degree: training, validation and ${curves.cv.length ? 'cross-validation' : ''} error; cross-validation picks degree ${curves.cvBest}`}>
        {ticks.map(t => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(10 ** t)} y2={y(10 ** t)} className="line-chart-grid" />
            <text x={PAD.left - 6} y={y(10 ** t)} className="line-chart-tick" textAnchor="end" dominantBaseline="middle">{t >= -2 && t <= 2 ? String(10 ** t) : `1e${t}`}</text>
          </g>
        ))}
        {Array.from({ length: MAX_DEGREE + 1 }, (_, d) => (
          <text key={d} x={x(d)} y={H - PAD.bottom + 13} className="line-chart-tick" textAnchor="middle">{d % 3 === 0 || d === MAX_DEGREE ? d : ''}</text>
        ))}
        <text x={PAD.left + 70} y={H - 2} className="line-chart-tick" textAnchor="middle">degree →</text>
        <line x1={PAD.left} x2={W - PAD.right} y1={y(noise * noise)} y2={y(noise * noise)} className="ms-noise" />
        <text x={W - PAD.right - 2} y={y(noise * noise) - 4} className="line-chart-tick" textAnchor="end">noise σ²</text>
        <line x1={x(curves.cvBest)} x2={x(curves.cvBest)} y1={PAD.top} y2={H - PAD.bottom} className="ms-cv-pick" />
        <line x1={x(degree)} x2={x(degree)} y1={PAD.top} y2={H - PAD.bottom} className="line-chart-marker" />
        {curves.cv.map((m, d) => (
          <line key={d} x1={x(d)} x2={x(d)} y1={y(m + curves.cvSe[d])} y2={y(Math.max(m - curves.cvSe[d], m / 1000))} className="ms-errbar" />
        ))}
        <path d={line(curves.train)} className="ms-line ms-train" />
        <path d={line(curves.validation)} className="ms-line ms-validation" />
        <path d={line(curves.cv)} className="ms-line ms-cv" />
        {curves.cv.map((m, d) => <circle key={d} cx={x(d)} cy={y(m)} r={2.5} className="ms-cv-dot" />)}
        {Array.from({ length: MAX_DEGREE + 1 }, (_, d) => (
          <rect key={d} x={x(d) - step / 2} y={PAD.top} width={step} height={H - PAD.top - PAD.bottom} className="ms-hit" onClick={() => onSelect(d)} />
        ))}
      </svg>
      <figcaption className="line-chart-legend">
        <span className="line-chart-key"><span className="line-chart-swatch ms-sw-train" />train</span>
        <span className="line-chart-key"><span className="line-chart-swatch ms-sw-validation" />validation</span>
        <span className="line-chart-key"><span className="line-chart-swatch ms-sw-cv" />k-fold CV ± 1 s.e.</span>
        <span className="line-chart-key"><span className="line-chart-swatch line-chart-swatch-dashed ms-sw-pick" />CV pick</span>
        <span className="line-chart-key"><span className="line-chart-swatch line-chart-swatch-dashed ms-sw-selected" />selected</span>
      </figcaption>
    </figure>
  );
}

export function ModelSelectionSection() {
  const [n, setN] = useState(30);
  const [noise, setNoise] = useState(0.25);
  const [seed, setSeed] = useState(1);
  const [degree, setDegree] = useState(3);
  const [logLambda, setLogLambda] = useState(-8);
  const [k, setK] = useState(5);
  const [heldOut, setHeldOut] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const lambda = logLambda <= -8 ? 0 : 10 ** logLambda;
  const fold = Math.min(heldOut, k - 1);

  const train = useMemo(() => makeRegression(n, noise, seededRandom(seed)), [n, noise, seed]);
  const validation = useMemo(() => makeRegression(n, noise, seededRandom(seed + VALIDATION_SEED)), [n, noise, seed]);
  const folds = useMemo(() => kFolds(n, k, seededRandom(seed + FOLD_SEED)), [n, k, seed]);
  const curves = useMemo(() => errorCurves(train, validation, folds, k, lambda), [train, validation, folds, k, lambda]);
  const fit = useMemo(() => fitPolynomial(train, degree, lambda), [train, degree, lambda]);
  const cv = useMemo(() => crossValidate(train, degree, folds, k, lambda), [train, degree, folds, k, lambda]);
  const bv = useMemo(() => biasVariance(degree, lambda, n, noise, seededRandom(seed * 7 + 3), { datasets: DATASETS }), [degree, lambda, n, noise, seed]);

  // Bias² and variance at every degree: 16 × 50 fits, recomputed in the background while sliders move.
  const settings = useDeferredValue({ lambda, n, noise, seed });
  const tradeOff = useMemo(() => {
    const parts = Array.from({ length: MAX_DEGREE + 1 }, (_, d) =>
      biasVariance(d, settings.lambda, settings.n, settings.noise, seededRandom(settings.seed * 7 + 3), { datasets: DATASETS, testPoints: 0, gridPoints: 100 }));
    return { bias2: parts.map(p => p.bias2), variance: parts.map(p => p.variance), total: parts.map(p => p.bias2 + p.variance + p.noise) };
  }, [settings]);

  const fitLayers = useMemo(() => [
    { xy: Float32Array.from({ length: 2 * n }, (_, i) => (i % 2 ? validation.y : validation.x)[i >> 1]), color: 'var(--series-orange)', shape: 'ring' as const, size: 3.5 },
    { xy: Float32Array.from({ length: 2 * n }, (_, i) => (i % 2 ? train.y : train.x)[i >> 1]), color: 'var(--series-blue)', size: 6 },
  ], [train, validation, n]);

  const fitOverlay = useMemo(() => (d: Draw) => {
    drawCurve(d, trueCurve, d.color('var(--text-muted)'), 2, [6, 5]);
    drawCurve(d, x => evaluate(fit, x), d.color('var(--series-green)'), 2.5);
  }, [fit]);

  const foldOverlay = useMemo(() => (d: Draw) => {
    const { ctx } = d;
    drawCurve(d, trueCurve, d.color('var(--text-muted)'), 1.5, [6, 5]);
    cv.fits.forEach((w, f) => { if (f !== fold) drawCurve(d, x => evaluate(w, x), d.color('color-mix(in srgb, var(--text) 30%, transparent)'), 1); });
    const colour = d.color(FOLD_COLORS[fold]);
    drawCurve(d, x => evaluate(cv.fits[fold], x), colour, 2.5);
    const outline = d.color('var(--bg)');
    for (let i = 0; i < n; i++) {
      const [px, py] = d.toPx(train.x[i], train.y[i]);
      ctx.beginPath();
      if (folds[i] === fold) {
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.strokeStyle = outline; ctx.lineWidth = 4; ctx.stroke();
        ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.stroke();
      } else {
        ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = d.color(`color-mix(in srgb, ${FOLD_COLORS[folds[i]]} 70%, transparent)`);
        ctx.fill();
      }
    }
  }, [cv, fold, folds, train, n]);

  const bvOverlay = useMemo(() => (d: Draw) => {
    const thin = d.color('color-mix(in srgb, var(--series-blue) 30%, transparent)');
    for (const w of bv.fits) drawCurve(d, x => evaluate(w, x), thin, 1);
    drawCurve(d, trueCurve, d.color('var(--text-muted)'), 2, [6, 5]);
    // The average fit, read off the grid.
    const { ctx } = d;
    ctx.beginPath();
    bv.grid.forEach((x, g) => {
      const [px, py] = d.toPx(x, Math.max(-50, Math.min(50, bv.meanFit[g])));
      if (g === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = d.color('var(--series-orange)');
    ctx.lineWidth = 3;
    ctx.stroke();
  }, [bv]);

  const sum = bv.bias2 + bv.variance + bv.noise;
  const scale = Math.max(sum, bv.measured);
  const pct = (v: number) => `${(100 * v) / scale}%`;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Points n <strong>{n}</strong>
            <input type="range" min={10} max={200} step={5} value={n} onChange={e => setN(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Noise σ <strong>{noise.toFixed(2)}</strong>
            <input type="range" min={0.05} max={0.6} step={0.05} value={noise} onChange={e => setNoise(Number(e.target.value))} />
          </label>
        </div>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 New sample</button>
        <span className="ml-auto">
          <ShowCodeButton code={selectionCode} title="model-selection/selection.ts" python={{ code: pythonCode, title: 'model-selection/selection.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Degree <strong>{degree}</strong>
            <input type="range" min={0} max={MAX_DEGREE} value={degree} onChange={e => setDegree(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Ridge λ <strong>{lambda ? lambda.toExponential(0) : 'off'}</strong>
            <input type="range" min={-8} max={0} step={0.5} value={logLambda} onChange={e => setLogLambda(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Folds k <strong>{k}</strong>
            <input type="range" min={2} max={10} value={k} onChange={e => setK(Number(e.target.value))} />
          </label>
        </div>
        <button type="button" className="btn btn-sm" disabled={degree === curves.cvBest} onClick={() => setDegree(curves.cvBest)}>Use the CV pick (degree {curves.cvBest})</button>
      </div>

      <div className="ml-grid ms-grid">
        <div className="demo-panel">
          <h3>Degree-{degree} fit</h3>
          <ScatterPlot layers={fitLayers} extent={EXTENT} overlay={fitOverlay}
            label={`${n} training points, ${n} validation points, the true curve and the degree-${degree} least-squares fit`} />
          <div className="ml-legend">
            <span><span className="ml-swatch" style={{ background: 'var(--series-blue)' }} />training points</span>
            <span><span className="ml-swatch ms-ring" />validation points</span>
            <span><span className="ml-swatch" style={{ background: 'var(--series-green)' }} />fit</span>
            <span><span className="ml-swatch" style={{ background: 'var(--text-muted)' }} />true curve (dashed)</span>
          </div>
          <div className="ml-stats">
            <span>train MSE</span><strong>{fmt(curves.train[degree])}</strong>
            <span>validation MSE</span><strong>{fmt(curves.validation[degree])}</strong>
            <span>{k}-fold CV MSE</span><strong>{fmt(curves.cv[degree])} ± {fmt(curves.cvSe[degree])}</strong>
          </div>
        </div>

        <div className="demo-panel">
          <h3>Error against degree</h3>
          <ErrorChart curves={curves} degree={degree} noise={noise} onSelect={setDegree} />
          <p className="caption">Log scale; click a degree to select it. Training error only falls. Validation and
          cross-validation error fall, then rise: the U. No model can beat the noise level σ² = {fmt(noise * noise)} on
          average. {k}-fold CV picks degree {curves.cvBest}.</p>
        </div>

        <div className="demo-panel">
          <h3>{k}-fold cross-validation, degree {degree}</h3>
          <div className="ms-stripes" role="group" aria-label="Folds: choose which one is held out">
            {Array.from({ length: k }, (_, f) => (
              <button key={f} type="button" className="ms-stripe" aria-pressed={fold === f} onClick={() => setHeldOut(f)}
                aria-label={`Hold out fold ${f + 1}: validation error ${fmt(cv.perFold[f])}`}>
                <span className="ms-stripe-label">{f + 1}</span>
                <svg viewBox={`0 0 ${n} 1`} preserveAspectRatio="none" aria-hidden="true">
                  {Array.from({ length: n }, (_, i) => (
                    <rect key={i} x={i + 0.08} y={0} width={0.84} height={1}
                      style={{ fill: folds[i] === f ? FOLD_COLORS[f] : 'color-mix(in srgb, var(--text) 14%, transparent)' }} />
                  ))}
                </svg>
                <span className="ms-stripe-err">{fmt(cv.perFold[f])}</span>
              </button>
            ))}
          </div>
          <p className="caption">Each row is one of the k rounds: its coloured cells (points sorted by x) are held out, the
          grey ones train the fit. Click a row.</p>
          <ScatterPlot layers={[]} extent={EXTENT} overlay={foldOverlay} size={300}
            label={`The training points coloured by fold, with fold ${fold + 1} held out and the fit trained without it`} />
          <p className="caption">Ringed: fold {fold + 1}, held out; its fit (coloured) never saw them and scores
          {' '}{fmt(cv.perFold[fold])} on them. Grey: the other {k - 1} rounds' fits. CV error = mean of the {k} =
          {' '}{fmt(cv.mean)}, at the cost of {k} fits per degree.</p>
        </div>

        <div className="demo-panel">
          <h3>Bias and variance, degree {degree}</h3>
          <ScatterPlot layers={[]} extent={EXTENT} overlay={bvOverlay} size={300}
            label={`${DATASETS} degree-${degree} fits to fresh datasets, their average and the true curve`} />
          <p className="caption">{DATASETS} fits, each to a fresh set of {n} points from the same process (thin), their
          average (thick orange) and the true curve (dashed).</p>
          <div className="ms-bars" aria-label="Bias squared, variance and noise against the measured expected test error">
            <span className="ms-bar-label">bias² + variance + noise</span>
            <div className="ms-bar">
              <span className="ms-seg ms-seg-bias" style={{ width: pct(bv.bias2) }} />
              <span className="ms-seg ms-seg-var" style={{ width: pct(bv.variance) }} />
              <span className="ms-seg ms-seg-noise" style={{ width: pct(bv.noise) }} />
            </div>
            <span className="ms-bar-label">measured test error</span>
            <div className="ms-bar"><span className="ms-seg ms-seg-measured" style={{ width: pct(bv.measured) }} /></div>
          </div>
          <div className="ml-stats">
            <span><span className="ml-swatch ms-seg-bias" />bias²</span><strong>{fmt(bv.bias2)}</strong>
            <span><span className="ml-swatch ms-seg-var" />variance</span><strong>{fmt(bv.variance)}</strong>
            <span><span className="ml-swatch ms-seg-noise" />noise σ²</span><strong>{fmt(bv.noise)}</strong>
            <span>sum</span><strong>{fmt(sum)}</strong>
            <span><span className="ml-swatch ms-seg-measured" />measured</span><strong>{fmt(bv.measured)}</strong>
          </div>
          <LineChart title="Bias squared, variance and their sum with the noise, against degree" logY xLabel="degree" markerX={degree} height={170}
            series={[
              { label: 'bias²', color: 'var(--series-red)', values: tradeOff.bias2 },
              { label: 'variance', color: 'var(--series-blue)', values: tradeOff.variance },
              { label: 'expected error', color: 'var(--text-muted)', values: tradeOff.total, dashed: true },
            ]} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Too simple, too flexible, and how to tell</h4>
        <p>A degree-0 polynomial is a flat line: it cannot follow the curve, and its error is high on the training points and
        on new ones alike. That is <strong>underfitting</strong>. Raise the degree and the fit bends to the data; by degree 3
        it follows the true curve closely. Keep going and it starts to bend to the noise too, threading through the training
        points and swinging wildly between them, above all near the ends where points are sparse. That is
        <strong> overfitting</strong>: the training error keeps falling, but the error on points the fit has not seen climbs.</p>
        <p>Training error can <em>only</em> fall with the degree, because each model contains the previous one (set the new
        coefficient to zero and you get the old fit back), and least squares picks the best of them on exactly those points.
        So it can never tell you when to stop. You need <strong>held-out data</strong>: points the fit never saw, like the
        orange rings here. And if you pick the degree by its error on the test set, that set has taken part in training, one
        choice at a time; its error is no longer an honest estimate, which is why practice keeps three sets: train, validation
        for choices, and a test set touched once at the end.</p>
        <p>Holding out points costs data. <strong>k-fold cross-validation</strong> reuses every point: split the data into k
        folds, fit k times, each time without one fold, and average the k errors on the folds left out. Every point is
        validated exactly once and used for training k − 1 times. The price is k fits per candidate (16 degrees × {k} folds
        = {16 * k} fits here) and a slightly pessimistic estimate, since each fit sees only (k − 1)/k of the data. The spread
        across folds shows how far to trust the differences: degrees whose CV errors are within a standard error of each
        other are hard to tell apart, and the "one-standard-error rule" takes the simplest of them.</p>
        <p>Why the U? Imagine drawing many training sets from the same process and fitting each. At a test point x:</p>
        <p className="formula"><code>E[(f̂(x) − y)²] = (E f̂(x) − f(x))² + E[(f̂(x) − E f̂(x))²] + σ²  =  bias² + variance + noise</code></p>
        <p>The last panel does exactly that experiment. A low degree gives fits that agree with each other but all miss the
        curve (bias); a high degree gives fits that are right on average but differ wildly from sample to sample (variance);
        no model removes the noise. The bars show the three parts measured separately and, next to them, the test error
        measured directly on fresh points: they agree up to Monte-Carlo error. With the defaults (30 points, σ = 0.25),
        degree 1 has bias² 0.033 and variance 0.009; degree 3 has bias² 0.0004 and the same variance, 0.009; degree 10 has
        a variance of 16, because a few of the 50 fits shoot off near the ends where a sample happens to have no points.
        More points shrink the variance; more noise inflates it.</p>
        <p>The <strong>ridge</strong> penalty λ Σ wⱼ² pulls the coefficients towards zero: the fits get stiffer and agree more
        with each other, so variance falls and bias rises. At degree 10, λ = 0.01 cuts the variance from 16 to 0.02 for a
        bias² of only 0.001; much larger λ flattens every fit towards a constant and the bias takes over. A well-chosen λ
        lets a flexible model behave like a simpler one, and the right-hand side of the validation U flattens. Neural networks have the same trade-off with far more
        parameters, handled with weight decay, dropout and early stopping: see the{' '}
        <a href="#/regularisation">Overfitting</a> lesson.</p>
      </div>
    </div>
  );
}
