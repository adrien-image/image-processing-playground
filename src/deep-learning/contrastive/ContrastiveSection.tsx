import { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import contrastiveCode from './contrastive.ts?raw';
import pythonCode from './contrastive.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import type { MnistData } from '../shared/mnist';
import { CLASS_COLORS, floatImage } from '../shared/render';
import { gatherRows, shuffled, smooth } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { augment, buildSimClr, disposeSimClr, embed, knnAccuracy, pca2, REPRESENTATION, trainStep, type SimClr } from './contrastive';
import '../shared/dl-shared.css';
import './contrastive.css';

const BATCH = 64;
const EVERY = 25;
const LABELLED = 500;
const TEST = 300;
const CHUNK = 100;
const SCATTER = 300;

export function ContrastiveSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Learn features from unlabelled digits: two augmentations of the same image should agree</span>
        <span className="ml-auto">
          <ShowCodeButton code={contrastiveCode} title="contrastive/contrastive.ts" python={{ code: pythonCode, title: 'contrastive/contrastive.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>A label for free: "these two are the same image"</h4>
        <p>Labelling data is expensive; images are not. <strong>Contrastive learning</strong> makes up a task that needs no
        labels: take a batch of N images, make two random augmentations of each (rotate, zoom, shift, add noise), and train
        the network so that each of the 2N views can pick out its partner among the others. With z the projected
        representation and sim the cosine similarity, the loss of view i with partner j is</p>
        <p className="formula"><code>{'ℓ(i, j) = −log( exp(sim(zᵢ, zⱼ)/τ) / Σ_{k ≠ i} exp(sim(zᵢ, z_k)/τ) )'}</code></p>
        <p>To find its partner after any augmentation, a representation must keep what the augmentations do not change, and
        throw away the exact position, angle and noise. For digits that is mostly which digit it is, so the digits form
        clusters although the network never saw a label: watch the scatter plot, a 2-D projection (PCA) of the
        representations of test digits, coloured by their true label.</p>
        <p>The labels are used only to measure the result: a <strong>5-nearest-neighbour</strong> vote among 500 labelled
        training digits, on the learned representation. Even a random encoder scores well above chance (random convolutions
        keep much of the image), and the score climbs as contrastive training goes on. Digits are an easy case: the same
        vote on raw pixels, shown for comparison, is already strong. On photos, where nearby pixels say little about what an
        image shows, features learned this way are far better than pixels, and SimCLR, MoCo and their successors pretrain
        large vision models like this before any labels are used. A smaller temperature τ makes the loss focus on the
        hardest negatives.</p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [steps, setSteps] = useState(400);
  const [temperature, setTemperature] = useState(0.5);
  const [losses, setLosses] = useState<number[]>([]);
  const [knn, setKnn] = useState<{ step: number; acc: number }[]>([]);
  const [scatter, setScatter] = useState<Float32Array | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const model = useRef<SimClr | null>(null);
  const abort = useRef<AbortController | null>(null);
  const stepRef = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // MNIST is stored digit by digit, so every subset is a random sample.
  const labelled = useMemo(() => {
    const rows = shuffled(data.train.count, seededRandom(8)).slice(0, LABELLED);
    return { images: gatherRows(data.train.images, 784, rows), labels: rows.map(r => data.train.labels[r]) };
  }, [data]);
  const testRows = useMemo(() => shuffled(data.test.count, seededRandom(9)).slice(0, TEST), [data]);
  const test = useMemo(() => ({ images: gatherRows(data.test.images, 784, testRows), labels: testRows.map(r => data.test.labels[r]) }), [data, testRows]);
  // Reference: the same vote on raw pixels.
  const rawKnn = useMemo(() => knnAccuracy(labelled.images, labelled.labels, test.images, test.labels, 784), [labelled, test]);
  const pairs = useMemo(() => {
    const random = seededRandom(5);
    const imgs = gatherRows(data.train.images, 784, shuffled(data.train.count, random).slice(0, 4));
    return [augment(imgs, 4, random), augment(imgs, 4, random)];
  }, [data]);

  const dispose = () => {
    abort.current?.abort();
    if (model.current) disposeSimClr(model.current);
    model.current = null;
  };
  useEffect(() => dispose, []);

  /** Representations of many images, a chunk at a time so the page stays responsive. */
  const embedAll = async (m: SimClr, images: Float32Array, count: number) => {
    const out = new Float32Array(count * REPRESENTATION);
    for (let s = 0; s < count; s += CHUNK) {
      const n = Math.min(CHUNK, count - s);
      out.set(embed(m, images.subarray(s * 784, (s + n) * 784), n), s * REPRESENTATION);
      await tf.nextFrame();
    }
    return out;
  };

  const measure = async (m: SimClr) => {
    const a = await embedAll(m, labelled.images, LABELLED), b = await embedAll(m, test.images, TEST);
    setKnn(k => [...k, { step: stepRef.current, acc: knnAccuracy(a, labelled.labels, b, test.labels, REPRESENTATION) }]);
    setScatter(pca2(b, TEST, REPRESENTATION));
  };

  const train = async () => {
    const controller = new AbortController();
    abort.current = controller;
    setError(null);
    model.current ??= buildSimClr();
    const m = model.current;
    const random = seededRandom(stepRef.current + 1);
    try {
      if (stepRef.current === 0) await measure(m);
      for (let s = 0; s < steps && !controller.signal.aborted; s++) {
        const rows = Array.from({ length: BATCH }, () => Math.floor(random() * data.train.count));
        const loss = trainStep(m, gatherRows(data.train.images, 784, rows), BATCH, random, temperature);
        stepRef.current++;
        setLosses(l => [...l, loss]);
        if (stepRef.current % EVERY === 0) {
          if (model.current !== m) break;
          await measure(m);
          setProgress(`step ${s + 1}/${steps}`);
        }
        await tf.nextFrame();
      }
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abort.current === controller) setProgress(null);
    }
  };

  const reset = () => {
    dispose();
    stepRef.current = 0;
    setLosses([]);
    setKnn([]);
    setScatter(null);
  };

  // The scatter plot of the representation's two principal components, coloured by label.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SCATTER * dpr;
    canvas.height = SCATTER * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1018';
    ctx.fillRect(0, 0, SCATTER, SCATTER);
    if (!scatter) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of scatter) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const span = hi - lo || 1;
    const px = (v: number) => 10 + ((v - lo) / span) * (SCATTER - 20);
    for (let i = 0; i < TEST; i++) {
      ctx.fillStyle = CLASS_COLORS[test.labels[i]];
      ctx.globalAlpha = 0.8;
      ctx.fillRect(px(scatter[2 * i]) - 1.5, px(scatter[2 * i + 1]) - 1.5, 3, 3);
    }
    ctx.globalAlpha = 1;
  }, [scatter, test]);

  const running = progress !== null;
  const latest = knn[knn.length - 1];

  return (
    <>
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={train} disabled={progress !== null}>
          {progress ? `⏳ ${progress}` : losses.length ? `▶ Train ${steps} more steps` : `▶ Train ${steps} steps (no labels)`}
        </button>
        {progress && <button type="button" className="btn btn-sm" onClick={() => abort.current?.abort()}>⏸ Stop</button>}
        <button type="button" className="btn btn-sm" onClick={reset} disabled={running}>↺ Reset</button>
        <div className="dl-sliders">
          <label className="dl-slider">Steps <strong>{steps}</strong>
            <input type="range" min={100} max={2000} step={100} value={steps} disabled={progress !== null} onChange={e => setSteps(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Temperature τ <strong>{temperature.toFixed(2)}</strong>
            <input type="range" min={0.05} max={1} step={0.05} value={temperature} disabled={progress !== null} onChange={e => setTemperature(Number(e.target.value))} />
          </label>
        </div>
        {latest && <span className="dl-status ml-auto">5-NN accuracy: {(latest.acc * 100).toFixed(1)}% (step {latest.step}) · raw pixels: {(rawKnn * 100).toFixed(1)}%</span>}
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="dl-grid cl-grid">
        <div className="demo-panel">
          <h3>Two augmentations of the same digits</h3>
          <div className="cl-pairs">
            {[0, 1, 2, 3].map(i => (
              <div key={i} className="cl-pair">
                <PixelCanvas image={floatImage(pairs[0].subarray(i * 784, (i + 1) * 784), 28, 28)} size={56} label="First augmented view" />
                <PixelCanvas image={floatImage(pairs[1].subarray(i * 784, (i + 1) * 784), 28, 28)} size={56} label="Second augmented view" />
              </div>
            ))}
          </div>
          <p className="caption">Each column is one positive pair; every other view in the batch is a negative.</p>
        </div>
        <div className="demo-panel">
          <h3>Test digits in the learned space (PCA)</h3>
          <canvas ref={canvasRef} className="cl-scatter" style={{ width: SCATTER, maxWidth: '100%', aspectRatio: '1' }} role="img"
            aria-label="Two principal components of the learned representations of 300 test digits, coloured by digit" />
          <div className="cl-legend">
            {CLASS_COLORS.map((c, d) => <span key={d}><i style={{ background: c }} />{d}</span>)}
          </div>
        </div>
        <div className="demo-panel">
          <h3>Loss and 5-NN accuracy</h3>
          <LineChart title="Contrastive loss, averaged over 10 steps" series={[{ label: 'NT-Xent loss', color: 'var(--series-red)', values: smooth(losses, 10) }]}
            xLabel="step / 10" height={150} />
          <LineChart title="5-nearest-neighbour accuracy on the learned representation" yMin={0} yMax={1}
            series={[{ label: '5-NN accuracy (labels used only here)', color: 'var(--series-green)', values: knn.map(k => k.acc) }]}
            xTickLabel={i => String(knn[i]?.step ?? i * EVERY)} xLabel="step" formatY={v => `${Math.round(v * 100)}%`} height={150} />
        </div>
      </div>
    </>
  );
}
