import { useCallback, useEffect, useMemo, useState, type PointerEvent } from 'react';
import knnCode from './knn.ts?raw';
import pythonCode from './knn.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import { DigitPicker } from '../../deep-learning/shared/DigitPicker';
import type { MnistData } from '../../deep-learning/shared/mnist';
import { floatImage } from '../../deep-learning/shared/render';
import { gatherRows, shuffled } from '../../deep-learning/shared/train';
import { useMnist } from '../../deep-learning/shared/useMnist';
import { makeClassification, SHAPE_LABELS, type Shape2D } from '../shared/datasets';
import { CLASS_COLORS, classMap, UNIT_EXTENT } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import { accuracyByK, decide, distanceRatio, nearest, uniformDistances, vote, type LabelledPoints, type Neighbours, type Weighting } from './knn';
import '../../deep-learning/shared/dl-shared.css';
import '../shared/ml-shared.css';
import './knn.css';

const SHAPES: Shape2D[] = ['blobs', 'moons', 'circles', 'xor', 'spiral'];
const N_TRAIN = 200, N_TEST = 500;
const KS = Array.from({ length: 26 }, (_, i) => 2 * i + 1); // 1, 3, …, 51
const PIXELS = 784;
const MNIST_KS = [1, 3, 5, 7, 9, 11, 13, 15];
const MNIST_TEST = 500;
const TRAIN_SIZES = [500, 2000, 6000];
const DIMS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
const CURSE_POINTS = 500, CURSE_QUERIES = 20;

const dataset = (shape: Shape2D, n: number, noise: number, seed: number): LabelledPoints => {
  const d = makeClassification(shape, n, noise, seededRandom(seed), 3);
  return { x: d.xy, labels: d.labels, dim: 2, classes: d.classes };
};

export function KnnSection() {
  const [weighting, setWeighting] = useState<Weighting>('uniform');
  const [showCode, setShowCode] = useState(false);
  const [trainSize, setTrainSize] = useState(2000);
  const mnist = useMnistKnn(trainSize);

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Voting:</span>
        {(['uniform', 'distance'] as Weighting[]).map(w => (
          <button key={w} type="button" aria-pressed={weighting === w} className={`btn btn-sm${weighting === w ? ' btn-primary' : ''}`}
            onClick={() => setWeighting(w)}>{w === 'uniform' ? 'Uniform (1 vote each)' : 'Distance-weighted (1/d)'}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={knnCode} title="knn/knn.ts" python={{ code: pythonCode, title: 'knn/knn.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <PlanePart weighting={weighting} />
      <MnistPart weighting={weighting} mnist={mnist} trainSize={trainSize} setTrainSize={setTrainSize} />
      <CursePart mnistRatio={mnist.ratio} />

      <div className="demo-explanation">
        <h4>Ask the neighbours</h4>
        <p>k-nearest neighbours has no training step: the model <em>is</em> the training set. To classify a point, find the k
        training points closest to it and let them vote. The vote fractions are an estimate of the class probabilities, which
        is what the background of the first plot shows:</p>
        <p className="formula"><code>p(c | x) ≈ (1/k) · #&#123;i ∈ N_k(x) : yᵢ = c&#125;      (distance-weighted: weights 1/dᵢ)</code></p>
        <p><strong>k = 1 memorises.</strong> Every training point is its own nearest neighbour, so the training accuracy is
        100% however noisy the data: the boundary bends around each noisy point and makes islands for it. The chart shows the
        usual picture: training accuracy falls as k grows, while test accuracy first rises (the vote averages the noise
        away) and then falls again when the neighbourhood is so wide that it swallows the other class. On Circles with noise 0.2,
        k = 1 gets 81.8% of the fresh points right, k = 9 86.6% and k = 51 only 71.6%, as the inner ring's votes are outnumbered
        by the outer ring's. Small k is a flexible,
        high-variance model, large k a rigid, high-bias one. With <strong>distance weighting</strong> a training point sits at
        distance 0 from itself and outvotes every other neighbour, so the training accuracy stays at 100% for every k: the
        train curve says nothing then, only the test curve does.</p>
        <p><strong>It works on raw pixels.</strong> Treat each 28 × 28 digit as a point in 784 dimensions and the same
        algorithm, with nothing learned, recognises handwritten digits. On the same 500 random test digits, 1-NN gets 82.2%
        right with 500 training digits, 88.2% with 2000 and 92.2% with all 6000: the more examples it remembers, the closer the
        nearest one is likely to be. Larger k hardly helps here (91.8% at k = 3 with 6000), and with few training digits it
        hurts, because the extra neighbours come from farther away. Its weaknesses show too: every
        prediction compares the query with every training image (millions of operations), the memory is the whole training
        set, and pixel distance knows nothing about shape, so a digit shifted by two pixels is "far" from itself.</p>
        <p><strong>The curse of dimensionality.</strong> In high dimensions, distances between random points concentrate:
        each squared distance is a sum of d independent terms, so its spread grows like √d while its mean grows like d. The
        nearest point is then barely nearer than the farthest, and "nearest" stops meaning "similar": the ratio of the two
        distances is 0.024 in 2-D and 0.90 in 1000-D. kNN survives on MNIST
        because digits are not random points of the 784-dimensional cube: they lie near a surface of far fewer dimensions
        (a stroke's thickness, slant, position…), and distances along it still discriminate. Measured: for a test digit and
        2000 training digits the nearest is at 0.37 of the farthest distance, against 0.89 for random points in the same 784
        dimensions. That is also why kNN often
        works better after reducing the dimension first (the PCA lesson) or on learned features.</p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Part 1: the plane */

function PlanePart({ weighting }: { weighting: Weighting }) {
  const [shape, setShape] = useState<Shape2D>('moons');
  const [noise, setNoise] = useState(0.2);
  const [k, setK] = useState(15);
  const [probe, setProbe] = useState<{ x: number; y: number; pinned: boolean } | null>(null);

  const train = useMemo(() => dataset(shape, N_TRAIN, noise, 1), [shape, noise]);
  const test = useMemo(() => dataset(shape, N_TEST, noise, 2), [shape, noise]);
  const map = useMemo(() => classMap((x, y) => vote(nearest(train.x, 2, [x, y], k), train.labels, train.classes, k, weighting), train.classes, 60), [train, k, weighting]);
  const curves = useMemo(() => ({ train: accuracyByK(train, train, KS, weighting), test: accuracyByK(train, test, KS, weighting) }), [train, test, weighting]);
  const at = (k - 1) / 2;

  const probed = useMemo(() => {
    if (!probe) return null;
    const nb = nearest(train.x, 2, [probe.x, probe.y], k);
    const votes = vote(nb, train.labels, train.classes, k, weighting);
    return { nb, votes, winner: decide(votes, nb, train.labels) };
  }, [probe, train, k, weighting]);

  const layers = useMemo(() => {
    const mask = new Uint8Array(N_TRAIN);
    probed?.nb.index.forEach(i => { mask[i] = 1; });
    return [
      { xy: train.x as Float32Array, labels: train.labels, size: 6 },
      ...(probed ? [{ xy: train.x as Float32Array, color: 'var(--text)', shape: 'ring' as const, size: 6, mask }] : []),
    ];
  }, [train, probed]);

  const overlay = useCallback(({ ctx, toPx, scale, color }: Draw) => {
    if (!probe || !probed) return;
    const [px, py] = toPx(probe.x, probe.y);
    ctx.strokeStyle = color('var(--text)');
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const i of probed.nb.index) { ctx.moveTo(px, py); ctx.lineTo(...toPx(train.x[2 * i], train.x[2 * i + 1])); }
    ctx.stroke();
    // The neighbourhood: a circle through the k-th nearest point.
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(px, py, probed.nb.dist[probed.nb.dist.length - 1] * scale, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color(CLASS_COLORS[probed.winner]);
    ctx.strokeStyle = color('var(--text)');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(px - 5, py - 5, 10, 10);
    ctx.fill();
    ctx.stroke();
  }, [probe, probed, train]);

  const hover = (e: PointerEvent<HTMLDivElement>) => {
    if (probe?.pinned || e.pointerType === 'touch') return;
    const canvas = e.currentTarget.querySelector('canvas');
    if (!canvas) return;
    const r = canvas.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const { x, y } = UNIT_EXTENT;
    setProbe({ x: x[0] + fx * (x[1] - x[0]), y: y[1] - fy * (y[1] - y[0]), pinned: false });
  };

  return (
    <>
      <h3 className="knn-part">1 · Neighbours in the plane</h3>
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {SHAPES.map(s => (
          <button key={s} type="button" aria-pressed={shape === s} className={`btn btn-sm${shape === s ? ' btn-primary' : ''}`}
            onClick={() => setShape(s)}>{SHAPE_LABELS[s]}</button>
        ))}
        <div className="dl-sliders">
          <label className="dl-slider">Noise <strong>{noise.toFixed(2)}</strong>
            <input type="range" min={0} max={0.5} step={0.01} value={noise} onChange={e => setNoise(Number(e.target.value))} />
          </label>
          <label className="dl-slider">k <strong>{k}</strong>
            <input type="range" min={1} max={51} step={2} value={k} onChange={e => setK(Number(e.target.value))} />
          </label>
        </div>
      </div>

      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Vote fractions of the {k} nearest</h3>
          <div className="knn-probe" onPointerMove={hover} onPointerLeave={() => { if (!probe?.pinned) setProbe(null); }}>
            <ScatterPlot layers={layers} map={map} overlay={overlay} size={380}
              onPick={(x, y) => setProbe(p => (p?.pinned && Math.hypot(p.x - x, p.y - y) < 0.05 ? null : { x, y, pinned: true }))}
              label={`${N_TRAIN} training points over the kNN class-probability map, with the neighbours of the probed location`} />
          </div>
          <VoteBar probed={probed} classes={train.classes} k={k} weighting={weighting} pinned={!!probe?.pinned} />
        </div>

        <div className="demo-panel">
          <h3>Accuracy against k</h3>
          <LineChart title="Training and test accuracy of kNN for k from 1 to 51" height={230} yMax={1}
            series={[
              { label: `train (${N_TRAIN} points)`, color: 'var(--series-blue)', values: curves.train },
              { label: `test (${N_TEST} fresh points)`, color: 'var(--series-orange)', values: curves.test },
            ]}
            xLabel="k" markerX={at} xTickLabel={i => String(KS[i])} formatY={v => `${Math.round(v * 100)}%`} />
          <div className="ml-stats">
            <span>k = {k}: train</span><strong>{(100 * curves.train[at]).toFixed(1)}%</strong>
            <span>k = {k}: test</span><strong>{(100 * curves.test[at]).toFixed(1)}%</strong>
            <span>best test k</span><strong>{KS[curves.test.indexOf(Math.max(...curves.test))]} ({(100 * Math.max(...curves.test)).toFixed(1)}%)</strong>
          </div>
          <p className="caption">At k = 1 the training accuracy is always 100%: each point is its own nearest neighbour. The
          test accuracy is what matters, and it is best at a moderate k.</p>
        </div>
      </div>
    </>
  );
}

function VoteBar({ probed, classes, k, weighting, pinned }: { probed: { nb: Neighbours; votes: Float64Array; winner: number } | null; classes: number; k: number; weighting: Weighting; pinned: boolean }) {
  if (!probed) return <p className="caption knn-hint">Hover over the plot (or tap it) to see the {k} neighbours of a location and their vote.</p>;
  return (
    <div className="knn-vote">
      <div className="knn-vote-bar" role="img" aria-label={`Vote: ${Array.from(probed.votes, (v, c) => `class ${c} ${Math.round(100 * v)}%`).join(', ')}`}>
        {Array.from(probed.votes, (v, c) => v > 0 && (
          <span key={c} style={{ width: `${100 * v}%`, background: CLASS_COLORS[c] }}>{v > 0.12 ? `${Math.round(100 * v)}%` : ''}</span>
        ))}
      </div>
      <p className="caption">
        {weighting === 'uniform' ? `${k} votes` : `${k} votes weighted by 1/distance`} over {classes} classes, won by
        <span className="ml-swatch knn-inline-swatch" style={{ background: CLASS_COLORS[probed.winner] }} />
        · {pinned ? 'pinned: click the square again to release' : 'click to pin'}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ Part 2: MNIST */

interface MnistRecord { index: number[]; dist: number[]; ratio: number; label: number }
interface MnistSample { images: Float32Array; labels: Uint8Array; testIndices: number[] }
interface MnistKnn { data: MnistData | null; error: string | null; sample: MnistSample | null; records: MnistRecord[]; ratio: number | null }

/** A random training sample, and the k nearest training digits of 500 random test digits, measured a few per frame. */
function useMnistKnn(trainSize: number): MnistKnn {
  const { data, error } = useMnist();
  const sample = useMemo<MnistSample | null>(() => {
    if (!data) return null;
    // The splits are stored digit by digit, so take random samples, never the first n.
    const rows = shuffled(data.train.count, seededRandom(11)).slice(0, trainSize);
    return {
      images: gatherRows(data.train.images, PIXELS, rows),
      labels: Uint8Array.from(rows, r => data.train.labels[r]),
      testIndices: shuffled(data.test.count, seededRandom(12)).slice(0, MNIST_TEST),
    };
  }, [data, trainSize]);
  const [run, setRun] = useState<{ sample: MnistSample | null; records: MnistRecord[] }>({ sample: null, records: [] });

  useEffect(() => {
    if (!data || !sample) return;
    let q = 0, id = 0;
    const step = () => {
      // As many test digits as fit in about 12 ms, then let the browser draw.
      const start = performance.now(), chunk: MnistRecord[] = [];
      while (q < sample.testIndices.length && performance.now() - start < 12) {
        const t = sample.testIndices[q++];
        const nb = nearest(sample.images, PIXELS, data.test.images, MNIST_KS[MNIST_KS.length - 1], t * PIXELS);
        chunk.push({ index: nb.index, dist: nb.dist, ratio: nb.dist[0] / nb.farthest, label: data.test.labels[t] });
      }
      setRun(r => ({ sample, records: r.sample === sample ? [...r.records, ...chunk] : chunk }));
      if (q < sample.testIndices.length) id = requestAnimationFrame(step);
    };
    id = requestAnimationFrame(step);
    return () => cancelAnimationFrame(id);
  }, [data, sample]);

  const records = run.sample === sample ? run.records : [];
  const ratio = records.length ? records.reduce((s, r) => s + r.ratio, 0) / records.length : null;
  return { data, error, sample, records, ratio };
}

function mnistAccuracy(records: MnistRecord[], labels: Uint8Array, k: number, weighting: Weighting): number {
  let right = 0;
  for (const r of records) {
    const nb = { index: r.index, dist: r.dist, farthest: 0 };
    if (decide(vote(nb, labels, 10, k, weighting), nb, labels) === r.label) right++;
  }
  return records.length ? right / records.length : 0;
}

function MnistPart({ weighting, mnist, trainSize, setTrainSize }: { weighting: Weighting; mnist: MnistKnn; trainSize: number; setTrainSize: (n: number) => void }) {
  const [k, setK] = useState(3);
  const [picked, setPicked] = useState(0);
  const { data, error, sample, records } = mnist;
  const done = records.length === MNIST_TEST;

  const curves = useMemo(() => {
    if (!sample) return null;
    const of = (w: Weighting) => MNIST_KS.map(kk => mnistAccuracy(records, sample.labels, kk, w));
    return { uniform: of('uniform'), distance: of('distance') };
  }, [records, sample]);

  const pickedResult = useMemo(() => {
    if (!data || !sample) return null;
    const nb = nearest(sample.images, PIXELS, data.test.images, MNIST_KS[MNIST_KS.length - 1], picked * PIXELS);
    return { nb, images: nb.index.map(i => floatImage(sample.images.subarray(i * PIXELS, (i + 1) * PIXELS), 28, 28)) };
  }, [data, sample, picked]);
  const queryImage = useMemo(() => (data ? floatImage(data.test.images.subarray(picked * PIXELS, (picked + 1) * PIXELS), 28, 28) : null), [data, picked]);

  const shownNb = pickedResult && { ...pickedResult.nb, index: pickedResult.nb.index.slice(0, k), dist: pickedResult.nb.dist.slice(0, k) };
  const votes = shownNb && sample ? vote(shownNb, sample.labels, 10, k, weighting) : null;
  const prediction = shownNb && votes && sample ? decide(votes, shownNb, sample.labels) : null;
  const accuracy = curves ? curves[weighting][MNIST_KS.indexOf(k)] : 0;

  return (
    <>
      <h3 className="knn-part">2 · Handwritten digits, pixel by pixel</h3>
      {error && <p className="error-banner">Could not load MNIST: {error}</p>}
      <div className="demo-controls">
        <span className="demo-label">Training digits:</span>
        {TRAIN_SIZES.map(n => (
          <button key={n} type="button" aria-pressed={trainSize === n} className={`btn btn-sm${trainSize === n ? ' btn-primary' : ''}`}
            onClick={() => setTrainSize(n)}>{n === 6000 ? 'all 6000' : n}</button>
        ))}
        <label className="dl-slider">k <strong>{k}</strong>
          <input type="range" min={1} max={15} step={2} value={k} onChange={e => setK(Number(e.target.value))} />
        </label>
        <span className="dl-status ml-auto">
          {!data ? 'loading MNIST…' : done ? `${MNIST_TEST} random test digits: ${(100 * accuracy).toFixed(1)}% right`
            : `measuring… ${records.length}/${MNIST_TEST} test digits, ${(100 * accuracy).toFixed(1)}% right so far`}
        </span>
      </div>
      {data && sample && (
        <div className="ml-grid">
          <div className="demo-panel">
            <h3>A test digit and its {k} nearest training digits</h3>
            <DigitPicker split={data.test} value={picked} onChange={setPicked} />
            <div className="knn-digits">
              <figure className="knn-query">
                {queryImage && <PixelCanvas image={queryImage} size={84} label={`Test digit, a ${data.test.labels[picked]}`} />}
                <figcaption>query ({data.test.labels[picked]})</figcaption>
              </figure>
              <div className="knn-neighbours">
                {pickedResult?.images.slice(0, k).map((img, r) => {
                  const label = sample.labels[pickedResult.nb.index[r]];
                  return (
                    <figure key={r} className={label === data.test.labels[picked] ? 'knn-match' : 'knn-miss'}>
                      <PixelCanvas image={img} size={50} label={`Neighbour ${r + 1}, a ${label}, at distance ${pickedResult.nb.dist[r].toFixed(2)}`} />
                      <figcaption>{label} · {pickedResult.nb.dist[r].toFixed(1)}</figcaption>
                    </figure>
                  );
                })}
              </div>
            </div>
            {prediction !== null && (
              <p className="caption">Vote → <strong className={prediction === data.test.labels[picked] ? 'knn-right' : 'knn-wrong'}>{prediction}</strong> (true
              label {data.test.labels[picked]}). Under each neighbour: its label · its Euclidean distance in pixel space (pixels in [0, 1]).</p>
            )}
          </div>
          <div className="demo-panel">
            <h3>Test accuracy against k</h3>
            {curves && (
              <LineChart title="kNN test accuracy on MNIST against k, for uniform and distance-weighted voting" height={230}
                series={[
                  { label: 'uniform', color: 'var(--series-blue)', values: curves.uniform },
                  { label: 'distance-weighted', color: 'var(--series-green)', values: curves.distance, dashed: true },
                ]}
                xLabel="k" markerX={MNIST_KS.indexOf(k)} xTickLabel={i => String(MNIST_KS[i])} formatY={v => `${(v * 100).toFixed(1)}%`} />
            )}
            <p className="caption">{trainSize} training digits (a random sample), {done ? MNIST_TEST : records.length} random test
            digits, raw pixels, nothing learned. Each test digit is compared with all {trainSize} training digits:
            {' '}{(trainSize * PIXELS / 1e6).toFixed(1)} million pixel differences per prediction.</p>
          </div>
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ Part 3: the curse of dimensionality */

function CursePart({ mnistRatio }: { mnistRatio: number | null }) {
  const [dimIndex, setDimIndex] = useState(4);
  const ratios = useMemo(() => {
    const random = seededRandom(21);
    return { dims: DIMS.map(d => distanceRatio(d, CURSE_POINTS, CURSE_QUERIES, random)), mnistDim: distanceRatio(PIXELS, CURSE_POINTS, CURSE_QUERIES, random) };
  }, []);
  const d = DIMS[dimIndex];
  const hist = useMemo(() => {
    const dist = uniformDistances(d, 2000, seededRandom(22 + d));
    const max = Math.max(...dist);
    return histogram(Array.from(dist, v => v / max), 40);
  }, [d]);

  return (
    <>
      <h3 className="knn-part">3 · The curse of dimensionality</h3>
      <div className="ml-grid">
        <div className="demo-panel">
          <h3>Nearest ÷ farthest distance</h3>
          <LineChart title="Ratio of the nearest to the farthest distance from a query, for uniform random points, against the dimension" height={230}
            yMin={0} yMax={1}
            series={[
              { label: `uniform random points (${CURSE_POINTS} in [0, 1]^d)`, color: 'var(--series-blue)', values: ratios.dims },
              ...(mnistRatio !== null ? [{ label: 'MNIST test digit vs training digits (d = 784)', color: 'var(--series-pink)', values: DIMS.map(() => mnistRatio), dashed: true }] : []),
            ]}
            xLabel="dimension d (log-spaced)" markerX={dimIndex} xTickLabel={i => String(DIMS[i])} formatY={v => v.toFixed(2)} />
          <div className="ml-stats">
            <span>d = 2</span><strong>{ratios.dims[1].toFixed(3)}</strong>
            <span>d = 1000</span><strong>{ratios.dims[DIMS.length - 1].toFixed(3)}</strong>
            <span>uniform, d = 784</span><strong>{ratios.mnistDim.toFixed(3)}</strong>
            <span>MNIST, d = 784</span><strong>{mnistRatio === null ? '…' : mnistRatio.toFixed(3)}</strong>
          </div>
          <p className="caption">Averaged over {CURSE_QUERIES} random queries. A ratio near 1 means the nearest neighbour is hardly
          nearer than the farthest point: distance no longer tells similar from dissimilar.</p>
        </div>
        <div className="demo-panel">
          <h3>Distances from one query, d = {d}</h3>
          <label className="dl-slider">Dimension d <strong>{d}</strong>
            <input type="range" min={0} max={DIMS.length - 1} step={1} value={dimIndex} onChange={e => setDimIndex(Number(e.target.value))} />
          </label>
          <Histogram counts={hist} label={`Histogram of the distances from a random query to 2000 random points in ${d} dimensions, divided by the largest`} />
          <p className="caption">Distances to 2000 uniform random points, divided by the largest. In 2-D they spread from 0 to 1;
          by d = 1000 they all pile up in a narrow band just below the maximum.</p>
        </div>
      </div>
    </>
  );
}

function histogram(values: number[], bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor(v * bins))]++;
  return counts;
}

function Histogram({ counts, label }: { counts: number[]; label: string }) {
  const W = 360, H = 170, pad = { l: 8, r: 8, t: 8, b: 22 };
  const max = Math.max(...counts), bw = (W - pad.l - pad.r) / counts.length;
  return (
    <svg className="knn-hist" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {counts.map((c, i) => {
        const h = (c / max) * (H - pad.t - pad.b);
        return <rect key={i} x={pad.l + i * bw + 0.5} y={H - pad.b - h} width={bw - 1} height={h} className="knn-hist-bar" />;
      })}
      <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} className="knn-hist-axis" />
      {[0, 0.5, 1].map(t => (
        <text key={t} x={pad.l + t * (W - pad.l - pad.r)} y={H - 6} className="knn-hist-tick" textAnchor={t === 0 ? 'start' : t === 1 ? 'end' : 'middle'}>{t === 1 ? '1 (farthest)' : t}</text>
      ))}
    </svg>
  );
}
