import { useEffect, useMemo, useState, type ReactNode } from 'react';
import boostingCode from './boosting.ts?raw';
import pythonCode from './boosting.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { makeRegression, trueCurve } from '../shared/datasets';
import { predictValue, splitRows } from '../trees/trees';
import { boostRound, startBoosting, type BoostOptions, type BoostState, type Data1D } from './boosting';
import '../shared/ml-shared.css';
import './boosting.css';

const N_TRAIN = 80;
const N_TEST = 200;
const MAX_ROUNDS = 300;
const ROUNDS_PER_FRAME = 2;
const GRID = Array.from({ length: 401 }, (_, i) => -1 + i / 200);
const W = 480, H = 300, PAD = 8;

function makeData(noise: number, seed: number): { train: Data1D; test: Data1D } {
  const d = makeRegression(N_TRAIN + N_TEST, noise, seededRandom(seed));
  const rows = splitRows(d.n, N_TEST / d.n, seededRandom(seed + 50));
  const pick = (r: number[]) => ({ x: Float64Array.from(r, i => d.x[i]), y: Float64Array.from(r, i => d.y[i]) });
  return { train: pick(rows.train), test: pick(rows.test) };
}

/** An SVG plot of x ∈ [−1, 1] against y ∈ [−range, range]; children get the coordinate maps. */
function Plot({ range, label, children }: { range: number; label: string; children: (px: (x: number) => number, py: (y: number) => number) => ReactNode }) {
  const px = (x: number) => PAD + ((x + 1) / 2) * (W - 2 * PAD);
  const py = (y: number) => H / 2 - (y / range) * (H / 2 - PAD);
  return (
    <svg className="boost-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <line className="boost-axis" x1={PAD} x2={W - PAD} y1={py(0)} y2={py(0)} />
      <line className="boost-axis" x1={px(0)} x2={px(0)} y1={PAD} y2={H - PAD} />
      <text className="boost-tick" x={px(0) + 4} y={PAD + 10}>{range.toPrecision(2)}</text>
      <text className="boost-tick" x={px(0) + 4} y={H - PAD - 2}>−{range.toPrecision(2)}</text>
      {children(px, py)}
    </svg>
  );
}

const linePath = (xs: ArrayLike<number>, ys: ArrayLike<number>, px: (x: number) => number, py: (y: number) => number, range: number) =>
  Array.from(xs, (x, i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(Math.max(-range, Math.min(range, ys[i]))).toFixed(1)}`).join('');

export function BoostingSection() {
  const [noise, setNoise] = useState(0.3);
  const [seed, setSeed] = useState(1);
  const [depth, setDepth] = useState(2);
  const [rate, setRate] = useState(0.1);
  const [maxRounds, setMaxRounds] = useState(MAX_ROUNDS);
  const [playing, setPlaying] = useState(false);
  // null: not boosted yet (F₀ only); every change of data or settings starts over.
  const [boosted, setBoosted] = useState<BoostState | null>(null);
  const [showCode, setShowCode] = useState(false);

  const { train, test } = useMemo(() => makeData(noise, seed), [noise, seed]);
  const fresh = useMemo(() => startBoosting(train, test, GRID), [train, test]);
  const state = boosted ?? fresh;
  const options: BoostOptions = useMemo(() => ({ depth, rate, minLeaf: 1 }), [depth, rate]);
  const rounds = state.rounds.length;
  const running = playing && rounds < maxRounds;

  const advance = (s: BoostState, k: number) => {
    let next = s;
    for (let i = 0; i < k && next.rounds.length < maxRounds; i++) next = boostRound(next, train, test, GRID, options);
    return next;
  };

  useEffect(() => {
    if (!running) return;
    const id = requestAnimationFrame(() => setBoosted(advance(state, ROUNDS_PER_FRAME)));
    return () => cancelAnimationFrame(id);
  });

  const restart = () => setBoosted(null);
  const truth = useMemo(() => GRID.map(trueCurve), []);

  // Panel 2: the residuals the newest tree was fitted to, and that tree.
  const newest = state.rounds[rounds - 1];
  const residuals = rounds ? state.residuals : Float64Array.from(train.y, y => y - state.f0);
  const treeFit = newest ? GRID.map(x => predictValue(newest.tree, [x])) : null;
  const rRange = Math.max(0.1, 1.15 * Math.max(...Array.from(residuals, Math.abs)));

  const bestTest = Math.min(...state.testMse), bestAt = state.testMse.indexOf(bestTest);
  const pad = (v: number[]) => [...v, ...new Array<number>(Math.max(0, maxRounds + 1 - v.length)).fill(NaN)];
  const stat = (v: number) => v.toFixed(4);

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <button type="button" className="btn btn-sm" onClick={() => setBoosted(advance(state, 1))} disabled={rounds >= maxRounds}>+1 Step one round</button>
        <button type="button" className="btn btn-sm btn-primary" aria-pressed={running} onClick={() => setPlaying(!running)} disabled={!running && rounds >= maxRounds}>
          {running ? '⏸ Pause' : '▶ Play'}
        </button>
        <button type="button" className="btn btn-sm" onClick={() => setBoosted(advance(state, maxRounds))} disabled={rounds >= maxRounds}>⏭ Run to {maxRounds}</button>
        <button type="button" className="btn btn-sm" onClick={() => { restart(); setPlaying(false); }}>↺ Reset</button>
        <button type="button" className="btn btn-sm" onClick={() => { setSeed(s => s + 1); restart(); }}>↻ New sample</button>
        <span className="dl-status">round {rounds}</span>
        <span className="ml-auto">
          <ShowCodeButton code={boostingCode} title="boosting/boosting.ts" python={{ code: pythonCode, title: 'boosting/boosting.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Noise σ <strong>{noise.toFixed(2)}</strong>
            <input type="range" min={0} max={0.5} step={0.01} value={noise} onChange={e => { setNoise(Number(e.target.value)); restart(); }} />
          </label>
          <label className="dl-slider">Tree depth <strong>{depth}</strong>
            <input type="range" min={1} max={4} value={depth} onChange={e => { setDepth(Number(e.target.value)); restart(); }} />
          </label>
          <label className="dl-slider">Learning rate η <strong>{rate.toFixed(2)}</strong>
            <input type="range" min={0.05} max={1} step={0.05} value={rate} onChange={e => { setRate(Number(e.target.value)); restart(); }} />
          </label>
          <label className="dl-slider">Rounds <strong>{maxRounds}</strong>
            <input type="range" min={10} max={MAX_ROUNDS} step={10} value={maxRounds} onChange={e => setMaxRounds(Number(e.target.value))} />
          </label>
        </div>
      </div>

      <div className="ml-grid boost-grid">
        <div className="demo-panel">
          <h3>Data and the ensemble after {rounds} round{rounds === 1 ? '' : 's'}</h3>
          <Plot range={1.6} label={`Training and test points, the true curve and the boosted prediction after ${rounds} rounds`}>
            {(px, py) => (
              <>
                {Array.from(test.x, (x, i) => <circle key={`t${i}`} className="boost-test" cx={px(x)} cy={py(test.y[i])} r={2.5} />)}
                {Array.from(train.x, (x, i) => <circle key={`p${i}`} className="boost-train" cx={px(x)} cy={py(train.y[i])} r={3.5} />)}
                <path className="boost-truth" d={linePath(GRID, truth, px, py, 1.6)} />
                <path className="boost-model" d={linePath(GRID, state.gridF, px, py, 1.6)} />
              </>
            )}
          </Plot>
          <div className="ml-legend">
            <span><span className="ml-swatch boost-sw-train" />training point</span>
            <span><span className="ml-swatch boost-sw-test" />test point</span>
            <span><span className="ml-swatch boost-sw-truth" />true curve</span>
            <span><span className="ml-swatch boost-sw-model" />F<sub>{rounds}</sub>(x)</span>
          </div>
        </div>

        <div className="demo-panel">
          <h3>{newest ? `Residuals and the tree of round ${rounds}` : 'Residuals of the constant model'}</h3>
          <Plot range={rRange} label={newest ? `The residuals before round ${rounds} and the regression tree fitted to them` : 'The residuals of the constant model F0'}>
            {(px, py) => (
              <>
                {Array.from(train.x, (x, i) => <circle key={i} className="boost-train" cx={px(x)} cy={py(residuals[i])} r={3.5} />)}
                {treeFit && <path className="boost-tree" d={linePath(GRID, treeFit, px, py, rRange)} />}
                {treeFit && newest && <path className="boost-tree-scaled" d={linePath(GRID, treeFit.map(v => newest.rate * v), px, py, rRange)} />}
              </>
            )}
          </Plot>
          <div className="ml-legend">
            <span><span className="ml-swatch boost-sw-train" />residual y − F<sub>{Math.max(0, rounds - 1)}</sub>(x)</span>
            <span><span className="ml-swatch boost-sw-tree" />tree h<sub>{Math.max(1, rounds)}</sub> (depth {depth}, {2 ** depth} leaves at most)</span>
            <span><span className="ml-swatch boost-sw-scaled" />η·h, what is added</span>
          </div>
          <p className="caption">{newest ? 'The vertical scale follows the residuals: they shrink as the ensemble improves.' : 'Step one round to fit the first tree to these residuals.'}</p>
        </div>

        <div className="demo-panel">
          <h3>Mean squared error against rounds</h3>
          <LineChart title="Train and test mean squared error after each round, with the noise variance" xLabel="round" yMin={0} height={240}
            markerX={bestAt} formatY={v => v.toFixed(2)}
            series={[
              { label: 'train', color: 'var(--series-orange)', values: pad(state.trainMse), dashed: true },
              { label: 'test', color: 'var(--series-blue)', values: pad(state.testMse) },
              { label: 'noise σ² (best possible test MSE)', color: 'var(--text-muted)', values: new Array<number>(maxRounds + 1).fill(noise * noise), dashed: true },
            ]} />
          <div className="ml-stats">
            <span>train MSE</span><strong>{stat(state.trainMse[rounds])}</strong>
            <span>test MSE</span><strong>{stat(state.testMse[rounds])}</strong>
            <span>best test MSE (marker)</span><strong>{stat(bestTest)} @ {bestAt}</strong>
            <span>noise variance σ²</span><strong>{stat(noise * noise)}</strong>
          </div>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Gradient descent, one tree at a time</h4>
        <p>Boosting builds a model as a sum of weak learners, here regression trees of depth 1–4. It starts from the best constant,
        F₀ = mean(y), then repeats: compute the <strong>residuals</strong> r = y − F, fit a small tree to them, and add a fraction η of
        that tree to the model:</p>
        <p className="formula"><code>Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x),   hₘ ≈ argmin_h Σᵢ (rᵢ − h(xᵢ))²,   rᵢ = yᵢ − Fₘ₋₁(xᵢ)</code></p>
        <p>Why residuals? For the loss L = ½ Σ (y_i − F(x_i))², the derivative with respect to the prediction F(x_i) is −(y_i − F(x_i)):
        the residuals are the <strong>negative gradient</strong>. Each round is a step of gradient descent, not on weights but on the
        function itself, with the tree as the best piecewise-constant approximation of the gradient that generalises to new x, and η as
        the step size. Other losses (absolute error, log loss for classification) only change what "residual" means.</p>
        <p>Since each leaf predicts the mean residual of its points, every round lowers the training error (by n·r̄²·η(2 − η) per
        leaf), and with enough rounds the ensemble passes through every training point. The test error does not follow. On the default
        sample (80 training points, σ = 0.3, so no model can beat a test MSE of σ² = 0.09), depth 2 with η = 0.1 is best after 33 rounds
        (test MSE 0.110) and ends at 0.130 after 300, its prediction a jagged staircase chasing single noisy points. Depth 4 with η = 1
        is best after a single round (0.123) and ends at 0.151. <strong>Shrinkage</strong> makes each tree a small correction, so the
        noise is absorbed slowly: stumps with η = 0.05 need 226 rounds to reach their best, 0.108, and are still there at round 300. That
        is why practical boosting uses a small η, many rounds and <em>early stopping</em> on a validation set (the marker on the
        chart).</p>
        <p>Compare with the random forest of the previous lesson: a forest grows deep, low-bias, high-variance trees <em>in parallel</em>{' '}
        and averages them to cut the variance; boosting adds shallow, high-bias trees <em>in sequence</em>, each correcting what the
        others still get wrong, to cut the bias. XGBoost, LightGBM and CatBoost are industrial versions of exactly this loop (with
        second-order steps, regularised leaves, histogram splits and row/feature subsampling), and they still win most competitions on
        tabular data.</p>
      </div>
    </div>
  );
}
