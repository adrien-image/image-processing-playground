import { useEffect, useMemo, useRef, useState } from 'react';
import tsneCode from './tsne.ts?raw';
import pythonCode from './tsne.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { useMnist } from '../../deep-learning/shared/useMnist';
import { seededRandom } from '../../lib/random';
import { fitPca, project } from '../pca/pca';
import { DigitLegend, DigitMap } from '../pca/DigitMap';
import { sameDigitNeighbours, sampleDigits } from '../pca/digits';
import { affinities, nearestNeighbours, neighbourhoodKept, squaredDistances, Tsne } from './tsne';
import '../shared/ml-shared.css';
import './tsne.css';

/** Digits embedded, dimensions kept by the PCA pre-processing, iterations run. */
const N = 800;
const DIMS = 50;
const MAX_ITERATIONS = 1000;
const EARLY = 250;
/** Milliseconds of optimisation per animation frame. */
const BUDGET_MS = 12;
/** Neighbours compared by the "neighbourhood kept" score. */
const KEPT_K = 10;
const pct = (v: number) => `${(100 * v).toFixed(0)}\u00a0%`;

interface Prepared {
  /** The digits on their first DIMS principal components (n × DIMS), and on the first two. */
  Z: Float32Array;
  pcaXY: Float32Array;
  labels: Uint8Array;
  highNeighbours: Int32Array;
  pcaScores: { sameDigit: number; kept: number };
  highSameDigit: number;
}

interface Scores {
  sameDigit: number;
  kept: number;
}

interface Frame {
  key: string;
  Y: Float32Array;
  iteration: number;
  kl: number;
  history: number[];
  scores: Scores;
}

/** The running optimisation, outside React state (it is mutated every frame). */
interface Run {
  key: string;
  tsne: Tsne;
  history: number[];
  scores: Scores | null;
  scoredAt: number;
}

function scoreMap(Y: ArrayLike<number>, prep: Prepared): Scores {
  return {
    sameDigit: sameDigitNeighbours(Y, prep.labels, 5).overall,
    kept: neighbourhoodKept(prep.highNeighbours, Y, 2, N, KEPT_K),
  };
}

export function TsneSection() {
  const { data, error } = useMnist();
  const [prep, setPrep] = useState<Prepared | null>(null);
  const [perplexity, setPerplexity] = useState(30);
  const [draft, setDraft] = useState(30);
  const [seed, setSeed] = useState(1);
  const [version, setVersion] = useState(0);
  const [running, setRunning] = useState(true);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [showCode, setShowCode] = useState(false);
  const runRef = useRef<Run | null>(null);

  const sample = useMemo(() => (data ? sampleDigits(data.train, N, seededRandom(11)) : null), [data]);

  // Pre-processing, as is usual for t-SNE: reduce the 784 pixels to 50 principal components
  // (one subspace-iteration step per animation frame).
  useEffect(() => {
    if (!sample) return;
    const steps = fitPca(sample.images, N, 784, DIMS, { random: seededRandom(12), iterations: 6 });
    let id = 0;
    const tick = () => {
      const s = steps.next();
      if (!s.done) { id = requestAnimationFrame(tick); return; }
      const pca = s.value;
      const Z = project(pca, sample.images, N, Array.from({ length: DIMS }, (_, i) => i));
      const pcaXY = project(pca, sample.images, N, [0, 1]);
      const highNeighbours = nearestNeighbours(squaredDistances(Z, N, DIMS), N, KEPT_K);
      const partial = { Z, pcaXY, labels: sample.labels, highNeighbours };
      setPrep({
        ...partial,
        pcaScores: { sameDigit: sameDigitNeighbours(pcaXY, sample.labels, 5).overall, kept: neighbourhoodKept(highNeighbours, pcaXY, 2, N, KEPT_K) },
        highSameDigit: sameDigitHigh(highNeighbours, sample.labels),
      });
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [sample]);

  const P = useMemo(() => (prep ? affinities(prep.Z, N, DIMS, perplexity) : null), [prep, perplexity]);
  const key = `${perplexity}:${seed}:${version}`;

  // A few iterations per animation frame, within a time budget, until MAX_ITERATIONS.
  useEffect(() => {
    if (!running || !P || !prep) return;
    const id = requestAnimationFrame(() => {
      let run = runRef.current;
      if (!run || run.key !== key) {
        run = { key, tsne: new Tsne(P, N, { random: seededRandom(seed) }), history: [], scores: null, scoredAt: -Infinity };
        runRef.current = run;
      }
      const { tsne } = run;
      const start = performance.now();
      do {
        tsne.step();
        run.history.push(tsne.kl);
      } while (performance.now() - start < BUDGET_MS && tsne.iteration < MAX_ITERATIONS);
      const done = tsne.iteration >= MAX_ITERATIONS;
      // Scoring the map costs a few milliseconds: every 20 iterations is plenty.
      if (!run.scores || done || tsne.iteration - run.scoredAt >= 20) {
        run.scores = scoreMap(tsne.Y, prep);
        run.scoredAt = tsne.iteration;
      }
      setFrame({ key, Y: Float32Array.from(tsne.Y), iteration: tsne.iteration, kl: tsne.kl, history: run.history.slice(), scores: run.scores });
      if (done) setRunning(false);
    });
    return () => cancelAnimationFrame(id);
  }, [running, P, prep, key, seed, frame]);

  const current = frame?.key === key ? frame : null;
  const finished = (current?.iteration ?? 0) >= MAX_ITERATIONS;

  const restart = () => { setVersion(v => v + 1); setRunning(true); };
  const commitPerplexity = () => {
    if (draft === perplexity) return;
    setPerplexity(draft);
    setRunning(true);
  };

  const status = !data ? (error ? '' : 'Loading MNIST…')
    : !prep ? `Reducing ${N} digits to ${DIMS} principal components…`
    : !current ? 'Starting…'
    : `iteration ${current.iteration}${current.iteration < EARLY ? ' · early exaggeration (P × 12)' : ''} · KL(P ‖ Q) = ${current.kl.toFixed(3)}`;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        {finished ? (
          <button type="button" className="btn btn-sm btn-primary" disabled={!prep} onClick={restart}>↺ Run again</button>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" disabled={!prep} onClick={() => setRunning(r => !r)}>{running ? '⏸ Pause' : '▶ Resume'}</button>
        )}
        <button type="button" className="btn btn-sm" disabled={!prep} onClick={restart}>↺ Restart</button>
        <button type="button" className="btn btn-sm" disabled={!prep} onClick={() => { setSeed(s => s + 1); setRunning(true); }}>🎲 New seed ({seed})</button>
        <label className="dl-slider">perplexity <strong>{draft}</strong>
          <input type="range" min={5} max={100} value={draft} onChange={e => setDraft(Number(e.target.value))}
            onPointerUp={commitPerplexity} onKeyUp={commitPerplexity} onBlur={commitPerplexity} />
        </label>
        <span className="ml-auto">
          <ShowCodeButton code={tsneCode} title="tsne/tsne.ts" python={{ code: pythonCode, title: 'tsne/tsne.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <span className="dl-status" aria-live="polite">{status}</span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {prep && (
        <div className="ml-grid">
          <div className="demo-panel">
            <h3>t-SNE map of {N} digits</h3>
            {current
              ? <DigitMap xy={current.Y} labels={prep.labels} label={`t-SNE embedding of ${N} MNIST digits at iteration ${current.iteration}, coloured by digit`} />
              : <div className="tsne-placeholder muted">…</div>}
            <DigitLegend />
            <p className="caption">Perplexity {perplexity}, seed {seed}. The plot zooms to fit the points, so the scale changes as the map grows.</p>
          </div>
          <div className="demo-panel">
            <h3>PCA of the same digits (components 1 and 2)</h3>
            <DigitMap xy={prep.pcaXY} labels={prep.labels} label={`The same ${N} digits projected on their first two principal components`} />
            <DigitLegend />
            <p className="caption">The best linear 2-D view: the two directions of largest variance.</p>
          </div>
          <div className="demo-panel">
            <h3>How much neighbourhood each map keeps</h3>
            <div className="ml-stats tsne-scores">
              <span>score</span>
              <strong>t-SNE</strong>
              <strong>PCA 2-D</strong>
              <span>5 nearest neighbours are the same digit</span>
              <strong>{current ? pct(current.scores.sameDigit) : '–'}</strong>
              <strong>{pct(prep.pcaScores.sameDigit)}</strong>
              <span>{KEPT_K} nearest neighbours in {DIMS}-D still among the {KEPT_K} nearest</span>
              <strong>{current ? pct(current.scores.kept) : '–'}</strong>
              <strong>{pct(prep.pcaScores.kept)}</strong>
            </div>
            <p className="caption">For reference, in the {DIMS}-dimensional space itself {pct(prep.highSameDigit)} of the 5 nearest neighbours are the
            same digit: that is the most a map could keep.</p>
            <LineChart title="KL divergence between P and Q over the iterations" height={170} xLabel="iteration"
              series={[{ label: 'KL(P ‖ Q)', color: 'var(--series-blue)', values: current?.history ?? [] }]}
              markerX={EARLY} formatY={v => v.toFixed(1)} />
            <p className="caption">The marker is the end of early exaggeration: P is multiplied by 12 before it, so the true KL
            barely moves while the clusters form; it jumps when the exaggeration stops, then falls quickly.</p>
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Keep the neighbours close</h4>
        <p>PCA keeps the directions of largest variance, which preserves the big distances and squashes everything else onto a
        plane: on the right, most digits pile up in the middle. t-SNE gives up on distances altogether and tries to keep only
        <strong> who is next to whom</strong>. For every digit it turns the distances to the others into a probability of being
        neighbours, with a Gaussian whose width σᵢ is tuned per point; then it moves 2-D points around until their own neighbour
        probabilities match, by gradient descent on the Kullback–Leibler divergence:</p>
        <p className="formula"><code>pⱼ|ᵢ ∝ exp(−‖xᵢ − xⱼ‖² / 2σᵢ²),  qᵢⱼ ∝ (1 + ‖yᵢ − yⱼ‖²)⁻¹,  KL(P ‖ Q) = Σ pᵢⱼ log(pᵢⱼ / qᵢⱼ)</code></p>
        <p><strong>Why a Student t in the map.</strong> There is far more room around a point in 50 dimensions than in 2: a digit can
        have many neighbours at the same moderate distance, which a plane cannot hold (the <strong>crowding problem</strong>). The
        heavy-tailed t distribution lets moderately far points sit much further apart in the map at little cost, which opens up
        space between groups. The older SNE used a Gaussian in the map too, and squeezed the groups together in
        the centre.</p>
        <p><strong>Perplexity</strong> is the effective number of neighbours each point pays attention to: σᵢ is found by binary
        search so that the entropy of pⱼ|ᵢ equals log(perplexity), which gives small σ in dense regions and large σ in sparse
        ones. Low perplexity (5) shatters the map into many small clumps; high perplexity (100) looks at more global structure
        and gives rounder, more connected groups. Change the slider and watch the map rebuild.</p>
        <p><strong>What not to read from the picture.</strong> Press 🎲 New seed: the clusters come back, but in different places and
        orientations, so their positions relative to each other mean little. The sizes of clusters are not meaningful either
        (t-SNE expands dense groups and shrinks sparse ones, through the per-point σ), and nor are the gaps between them: a wide
        gap does not mean two digits are more different than a narrow one. The axes have no units. t-SNE is for looking at which
        points are grouped together, not for measuring distances, and a cluster on a t-SNE map is a hypothesis to check, not a
        proof. It also has no formula for placing a new point: run it again with the point included.</p>
        <p><strong>Cost.</strong> Every iteration compares all {(N * (N - 1) / 2).toLocaleString('en-GB')} pairs, so this exact version is limited to a few
        thousand points. Barnes–Hut and FFT-accelerated variants (and UMAP, a close cousin) handle millions.</p>
      </div>
    </div>
  );
}

/** Same-digit fraction of the k nearest neighbours already computed in the high-dimensional space (first 5 of them). */
function sameDigitHigh(neighbours: Int32Array, labels: Uint8Array): number {
  const k = neighbours.length / labels.length;
  let same = 0;
  for (let i = 0; i < labels.length; i++) for (let a = 0; a < 5; a++) if (labels[neighbours[i * k + a]] === labels[i]) same++;
  return same / (5 * labels.length);
}
