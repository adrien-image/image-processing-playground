import { useEffect, useMemo, useState } from 'react';
import gmmCode from './gmm.ts?raw';
import pythonCode from './gmm.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { covarianceEllipse } from '../shared/linalg';
import { CLASS_COLORS, classMap } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import type { Points2D } from '../shared/datasets';
import {
  ADDED, agreement, assignNearest, bicCurve, DATA_LABELS, eStep, hardLabels, inertia, initialCentres, makeMixtureData, mStep, nearest,
  objective, posterior, startMixture, updateCentres, type Mixture, type MixtureData, type Point,
} from './gmm';
import '../shared/ml-shared.css';
import './gmm.css';

const N = 300;
const DATA_SEED = 2;
const TRUE_K = 3;
const MAX_ITERATIONS = 300;
const TICK_MS = 40;

type InitMethod = 'kmeans++' | 'random';

/** Both algorithms, advanced in lockstep: E-step ↔ assignment step, M-step ↔ centre update. */
interface Fit {
  mix: Mixture;
  /** Responsibilities from the latest E-step (null before the first one). */
  resp: Float64Array | null;
  next: 'E' | 'M';
  /** The objective after 0, 1, 2, … full EM steps. */
  history: number[];
  centres: Point[];
  labels: Uint8Array | null;
  iteration: number;
  converged: boolean;
}

function freshFit(xy: Float32Array, centres: Point[]): Fit {
  const mix = startMixture(xy, centres);
  return { mix, resp: null, next: 'E', history: [objective(xy, mix)], centres, labels: null, iteration: 0, converged: false };
}

function halfStep(xy: Float32Array, f: Fit): Fit {
  if (f.next === 'E' || !f.resp || !f.labels) {
    return { ...f, resp: eStep(xy, f.mix).resp, labels: assignNearest(xy, f.centres), next: 'M' };
  }
  const mix = mStep(xy, f.resp, f.mix.length);
  const value = objective(xy, mix), prev = f.history[f.history.length - 1];
  return {
    ...f, mix, centres: updateCentres(xy, f.labels, f.centres), next: 'E', history: [...f.history, value],
    iteration: f.iteration + 1, converged: Math.abs(value - prev) < 1e-7 * Math.abs(prev),
  };
}

function rgbOf(css: string): [number, number, number] {
  const m = css.match(/[\d.]+/g);
  return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [128, 128, 128];
}

function drawMean(d: Draw, [x, y]: Point, colour: string) {
  const [px, py] = d.toPx(x, y);
  for (const [w, c] of [[5, d.color('var(--bg)')], [2.5, colour]] as const) {
    d.ctx.strokeStyle = c;
    d.ctx.lineWidth = w;
    d.ctx.beginPath();
    d.ctx.moveTo(px - 7, py - 7); d.ctx.lineTo(px + 7, py + 7); d.ctx.moveTo(px - 7, py + 7); d.ctx.lineTo(px + 7, py - 7);
    d.ctx.stroke();
  }
}

export function GmmSection() {
  const [kind, setKind] = useState<MixtureData>('stretched');
  const [K, setK] = useState(TRUE_K);
  const [method, setMethod] = useState<InitMethod>('kmeans++');
  const [seed, setSeed] = useState(1);
  const [points, setPoints] = useState<Points2D>(() => makeMixtureData('stretched', N, seededRandom(DATA_SEED)));
  const [start, setStart] = useState<Point[]>(() => initialCentres(points.xy, TRUE_K, seededRandom(1)));
  const [fit, setFit] = useState<Fit>(() => freshFit(points.xy, start));
  const [running, setRunning] = useState(false);
  const [showCode, setShowCode] = useState(false);

  /** New starting centres (and a fresh fit) after any change of data, K or initialisation. */
  const restart = (next: { kind?: MixtureData; K?: number; method?: InitMethod; seed?: number; points?: Points2D; keepStart?: boolean }) => {
    const k = next.K ?? K, m = next.method ?? method, s = next.seed ?? seed;
    const p = next.points ?? (next.kind ? makeMixtureData(next.kind, N, seededRandom(DATA_SEED)) : points);
    const centres = next.keepStart ? start : initialCentres(p.xy, k, seededRandom(s), m);
    if (next.kind) setKind(next.kind);
    setK(k); setMethod(m); setSeed(s); setPoints(p); setStart(centres);
    setFit(freshFit(p.xy, centres));
    setRunning(false);
  };

  // Run: half-steps at first so the alternation is visible, then whole iterations faster.
  useEffect(() => {
    if (!running) return;
    const id = setTimeout(() => {
      if (fit.converged || fit.iteration >= MAX_ITERATIONS) { setRunning(false); return; }
      let f = halfStep(points.xy, fit);
      if (f.iteration >= 10) for (let s = 0; s < 7 && !f.converged; s++) f = halfStep(points.xy, f);
      setFit(f);
    }, TICK_MS);
    return () => clearTimeout(id);
  }, [running, fit, points]);

  const addPoint = (x: number, y: number) => {
    const xy = Float32Array.from([...points.xy, x, y]);
    const labels = Uint8Array.from([...points.labels, ADDED]);
    restart({ points: { xy, labels, n: points.n + 1, classes: points.classes }, keepStart: true });
  };

  const gmmMap = useMemo(() => classMap((x, y) => posterior(x, y, fit.mix).resp, K, 64), [fit.mix, K]);
  const kmMap = useMemo(() => classMap((x, y) => nearest(x, y, fit.centres), K, 128), [fit.centres, K]);
  const bics = useMemo(() => bicCurve(points.xy, 6, seededRandom(1)), [points]);
  const bestK = bics.indexOf(Math.min(...bics)) + 1;

  const gmmOverlay = useMemo(() => (d: Draw) => {
    const { ctx } = d;
    const colours = CLASS_COLORS.map(c => d.color(c));
    const rgbs = colours.map(rgbOf);
    const outline = d.color('color-mix(in srgb, var(--bg) 85%, transparent)');
    const grey = d.color('var(--text-muted)');
    // Points: each mixes the component colours by its responsibilities.
    for (let i = 0; i < points.n; i++) {
      const [px, py] = d.toPx(points.xy[2 * i], points.xy[2 * i + 1]);
      let fill = grey;
      if (fit.resp) {
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < K; k++) {
          const w = fit.resp[i * K + k], c = rgbs[k % rgbs.length];
          r += w * c[0]; g += w * c[1]; b += w * c[2];
        }
        fill = `rgb(${r}, ${g}, ${b})`;
      }
      ctx.beginPath();
      ctx.arc(px, py, points.labels[i] === ADDED ? 4 : 2.8, 0, 2 * Math.PI);
      ctx.fillStyle = fill; ctx.strokeStyle = outline; ctx.lineWidth = 1;
      ctx.fill(); ctx.stroke();
    }
    // Ellipses at 1σ (solid) and 2σ (dashed), then the means.
    fit.mix.forEach((c, k) => {
      const [px, py] = d.toPx(c.mean[0], c.mean[1]);
      for (const s of [1, 2]) {
        const { rx, ry, angle } = covarianceEllipse(c.cov[0], c.cov[1], c.cov[2], s);
        ctx.beginPath();
        ctx.ellipse(px, py, Math.max(rx * d.scale, 0.5), Math.max(ry * d.scale, 0.5), -angle, 0, 2 * Math.PI);
        ctx.setLineDash(s === 2 ? [5, 4] : []);
        ctx.strokeStyle = outline; ctx.lineWidth = 4; ctx.stroke();
        ctx.strokeStyle = colours[k % colours.length]; ctx.lineWidth = 2; ctx.stroke();
      }
      ctx.setLineDash([]);
    });
    fit.mix.forEach((c, k) => drawMean(d, c.mean, colours[k % colours.length]));
  }, [points, fit, K]);

  const kmOverlay = useMemo(() => (d: Draw) => {
    fit.centres.forEach((c, k) => drawMean(d, c, d.color(CLASS_COLORS[k % CLASS_COLORS.length])));
  }, [fit.centres]);

  const kmLayers = useMemo(() => [fit.labels
    ? { xy: points.xy, labels: fit.labels, size: 5.6 }
    : { xy: points.xy, color: 'var(--text-muted)', size: 5.6 }], [points, fit.labels]);

  const gmmAgree = fit.resp && K === TRUE_K ? agreement(hardLabels(fit.resp, K), points.labels, K) : null;
  const kmAgree = fit.labels && K === TRUE_K ? agreement(fit.labels, points.labels, K) : null;
  const value = fit.history[fit.history.length - 1];
  const done = fit.converged || fit.iteration >= MAX_ITERATIONS;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {(Object.keys(DATA_LABELS) as MixtureData[]).map(d => (
          <button key={d} type="button" aria-pressed={kind === d} className={`btn btn-sm${kind === d ? ' btn-primary' : ''}`}
            onClick={() => restart({ kind: d })}>{DATA_LABELS[d]}</button>
        ))}
        <span className="muted">or click a plot to add points</span>
        <span className="ml-auto">
          <ShowCodeButton code={gmmCode} title="gmm/gmm.ts" python={{ code: pythonCode, title: 'gmm/gmm.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <label className="dl-slider">Components K <strong>{K}</strong>
          <input type="range" min={1} max={6} value={K} onChange={e => restart({ K: Number(e.target.value) })} />
        </label>
        <span className="demo-label">Start:</span>
        {(['kmeans++', 'random'] as const).map(m => (
          <button key={m} type="button" aria-pressed={method === m} className={`btn btn-sm${method === m ? ' btn-primary' : ''}`}
            onClick={() => restart({ method: m })}>{m === 'kmeans++' ? 'k-means++' : 'Random points'}</button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => restart({ seed: seed + 1 })}>🎲 New initialisation</button>
      </div>

      <div className="demo-controls">
        <button type="button" className="btn btn-sm" disabled={running || fit.next !== 'E' || done} onClick={() => setFit(halfStep(points.xy, fit))}>E-step</button>
        <button type="button" className="btn btn-sm" disabled={running || fit.next !== 'M'} onClick={() => setFit(halfStep(points.xy, fit))}>M-step</button>
        <button type="button" className="btn btn-sm btn-primary" disabled={done && !running} onClick={() => setRunning(r => !r)}>{running ? '⏸ Pause' : '▶ Run'}</button>
        <button type="button" className="btn btn-sm" onClick={() => restart({ keepStart: true })}>↺ Reset</button>
        <span className="dl-status ml-auto">
          iteration {fit.iteration}{fit.next === 'M' ? ' · E-step done, M-step next' : ''}{fit.converged ? ' · converged' : ''}
        </span>
      </div>

      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Gaussian mixture (EM)</h3>
          <ScatterPlot layers={[]} map={gmmMap} overlay={gmmOverlay} onPick={addPoint}
            label={`${points.n} points coloured by their responsibilities, with each component's 1σ and 2σ ellipse and mean`} />
          <p className="caption">Point colour mixes the components by rᵢₖ (grey: no E-step yet). Ellipses at 1σ (solid,
          39% of a component's mass) and 2σ (dashed, 86%); background: the most likely component.</p>
        </div>
        <div className="demo-panel">
          <h3>k-means (Lloyd), same starting centres</h3>
          <ScatterPlot layers={kmLayers} map={kmMap} overlay={kmOverlay} onPick={addPoint}
            label="The same points, assigned to the nearest k-means centre, over the Voronoi regions of the centres" />
          <p className="caption">Hard assignments to the nearest centre; the regions are the centres' Voronoi cells, whose
          borders are always straight lines halfway between two centres.</p>
        </div>
        <div className="gmm-side">
          <div className="demo-panel">
            <h3>Log-likelihood</h3>
            <LineChart title="The log-likelihood of the data after each EM iteration" xLabel="EM iteration"
              series={[{ label: 'log-likelihood (with the covariance floor)', color: 'var(--series-blue)', values: fit.history }]} height={170}
              formatY={v => v.toFixed(Math.abs(v) < 10 ? 1 : 0)} />
            <div className="ml-stats">
              <span>log-likelihood</span><strong>{value.toFixed(2)}</strong>
              <span>k-means inertia</span><strong>{fit.labels ? inertia(points.xy, fit.labels, fit.centres).toFixed(2) : '—'}</strong>
              <span>GMM matches true clusters</span><strong>{gmmAgree === null ? '—' : `${(100 * gmmAgree).toFixed(0)}%`}</strong>
              <span>k-means matches true clusters</span><strong>{kmAgree === null ? '—' : `${(100 * kmAgree).toFixed(0)}%`}</strong>
            </div>
            <p className="caption">{K === TRUE_K
              ? 'Agreement with the three clusters the data were drawn from, under the best relabelling (clicked points excluded).'
              : 'The data come from three clusters: set K = 3 to compare each method with the truth.'}</p>
          </div>
          <div className="demo-panel">
            <h3>Choosing K: BIC</h3>
            <LineChart title="Bayesian information criterion for K = 1 to 6, lower is better" xLabel="K" xStart={1} markerX={bestK - 1}
              series={[{ label: 'BIC = −2 log L + p ln n', color: 'var(--series-orange)', values: bics }]} height={170} />
            <p className="caption">Each K fitted from the best of three k-means++ starts. The minimum, marked, is at
            K = {bestK}{bestK === TRUE_K ? ', the true number of clusters' : ''}. More components always raise the likelihood; the
            penalty p ln n (p = 6K − 1 parameters) stops K growing for nothing.</p>
          </div>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Soft clusters with shapes</h4>
        <p>A Gaussian mixture is a story of how the data were made: choose a component k with probability πₖ, then draw the
        point from that component's Gaussian N(μₖ, Σₖ). Each component has its own weight, centre and full covariance, so it
        can be a long, thin, tilted cloud or a small, tight one. Fitting means finding the π, μ and Σ under which the data are
        most likely; the catch is that nobody tells us which component made which point.</p>
        <p><strong>Expectation–maximisation</strong> breaks that circle by alternating two easy problems. The
        <strong> E-step</strong> asks, with the current parameters, how responsible each component is for each point. That is
        just Bayes' rule, and every point's responsibilities sum to 1:</p>
        <p className="formula"><code>rᵢₖ = πₖ N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ | μⱼ, Σⱼ)</code></p>
        <p>The <strong>M-step</strong> then refits every component as if those soft assignments were known: weighted counts,
        means and covariances.</p>
        <p className="formula"><code>Nₖ = Σᵢ rᵢₖ,  πₖ = Nₖ / n,  μₖ = Σᵢ rᵢₖ xᵢ / Nₖ,  Σₖ = (Σᵢ rᵢₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ + λI) / Nₖ</code></p>
        <p>Neither step can lower the log-likelihood, so the curve only goes up; press E-step and M-step alternately to watch
        the colours change in one and the ellipses move in the other. The small λ = 0.001 is a covariance floor: without it a
        component could shrink onto a single point, where its density and the likelihood are infinite. (Click to add one lone
        point far from the rest and give it its own component to see the floor at work.) Formally EM climbs the
        log-likelihood minus a tiny penalty (λ/2)·tr Σₖ⁻¹, which is what the chart plots.</p>
        <p><strong>k-means</strong> is the same alternation with hard choices: assign each point wholly to its nearest centre,
        then move each centre to the mean of its points. It is the limit of a mixture with equal weights and equal, round
        covariances shrinking to zero, so it can only draw straight borders halfway between centres. On round blobs of equal
        size the two agree completely. On the <em>stretched clusters</em> k-means cuts across the stripes (with the default
        start it matches about 60% of the points to their true cluster, the mixture 100%), and with <em>unequal sizes</em> its
        halfway borders hand the big cluster's edge to the small ones (84% against 98%).</p>
        <p>EM only finds a <strong>local</strong> optimum, and the start matters: press New initialisation a few times on the
        unequal clusters and some starts end with one ellipse covering two clusters and a lower final log-likelihood (−282
        instead of −186). In practice one runs several starts and keeps the most likely. Choosing <strong>K</strong> cannot be
        done by likelihood alone, since more components always fit better; the BIC panel adds a penalty for every parameter and,
        on all three datasets here, has its minimum at the true K = 3.</p>
      </div>
    </div>
  );
}
