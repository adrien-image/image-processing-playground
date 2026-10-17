import { useEffect, useMemo, useRef, useState } from 'react';
import transferCode from './transfer.ts?raw';
import pythonCode from './transfer.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import type { MnistData } from '../shared/mnist';
import { floatImage } from '../shared/render';
import { useMnist } from '../shared/useMnist';
import {
  buildClassifier, buildFeatureExtractor, cloneExtractor, disposeClassifier, selectDigits, train,
  type Classifier, type Subset,
} from './transfer';
import '../shared/dl-shared.css';
import './transfer.css';

const OLD = [0, 1, 2, 3, 4];
const NEW = [5, 6, 7, 8, 9];
const SHOTS = [1, 2, 5, 10, 20, 50];
const STRATEGIES = [
  { key: 'scratch', label: 'from scratch', color: 'var(--series-red)', lr: 0.003 },
  { key: 'probe', label: 'linear probe (frozen features)', color: 'var(--series-blue)', lr: 0.003 },
  { key: 'finetune', label: 'fine-tune everything', color: 'var(--series-green)', lr: 0.0005 },
] as const;
type Strategy = (typeof STRATEGIES)[number]['key'];

interface Result { shots: number; curves: Record<Strategy, number[]> }

export function TransferSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Learn 0–4 from thousands of examples, then 5–9 from a handful</span>
        <span className="ml-auto">
          <ShowCodeButton code={transferCode} title="transfer/transfer.ts" python={{ code: pythonCode, title: 'transfer/transfer.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>Features are reusable</h4>
        <p>The first layers of a convolutional network learn edges and strokes; only the last layer decides which digit a
        combination of strokes is. A network trained on 0–4 never saw a 5, but it already knows the curves and bars a 5 is
        made of. With only a few labelled examples of the new digits there are three options:</p>
        <ul>
          <li><strong>From scratch</strong>: a fresh LeNet learns from the few examples alone. It fits them perfectly and
          generalises poorly: it has far more weights than examples.</li>
          <li><strong>Linear probe</strong>: keep the pretrained feature extractor frozen and train only a new last layer,
          84 × 5 + 5 = 425 weights. Freezing is just choosing which variables the optimizer may update.</li>
          <li><strong>Fine-tune</strong>: start from the pretrained weights and train everything, with a small learning rate so
          the features are adjusted rather than overwritten.</li>
        </ul>
        <p>The pretrained features help most when examples are scarcest: with one example per digit the linear probe scored
        about 13 points above the network trained from scratch in our runs. By five examples per digit the three are within a
        few points of each other, and with fifty they are equal. The gap is small here because digits are simple: a fresh
        network learns their strokes from a handful of examples. On photographs it is far larger, which is why most vision
        models today start from a backbone pretrained on millions of images and are adapted to a new task with a few
        hundred. With so few examples the numbers also depend on which ones were picked, so read differences of a few points as
        noise.</p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [pretrainAcc, setPretrainAcc] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  const [shots, setShots] = useState(5);
  const [results, setResults] = useState<Result[]>([]);
  const [live, setLive] = useState<Record<Strategy, number[]> | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pretrained = useRef<Classifier | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => {
    abort.current?.abort();
    if (pretrained.current) disposeClassifier(pretrained.current);
  }, []);

  const tests = useMemo(() => ({
    old: selectDigits(data.test, OLD, Infinity, seededRandom(1)),
    fresh: selectDigits(data.test, NEW, Infinity, seededRandom(2)),
  }), [data]);
  const examples: Subset = useMemo(() => selectDigits(data.train, NEW, shots, seededRandom(shots)), [data, shots]);

  const run = async (task: (signal: AbortSignal) => Promise<void>) => {
    const controller = new AbortController();
    abort.current = controller;
    setError(null);
    try { await task(controller.signal); } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally { setProgress(null); }
  };

  const pretrain = () => run(async signal => {
    const train03 = selectDigits(data.train, OLD, Infinity, seededRandom(3));
    const c = buildClassifier(buildFeatureExtractor(1), OLD.length, 0.002, false);
    setProgress('pretraining on 0–4: epoch 1/3');
    await train(c, train03, tests.old, OLD.length, 3, {
      random: seededRandom(4), signal,
      onEpoch: (e, acc) => { setProgress(`pretraining on 0–4: epoch ${e}/3 done`); setPretrainAcc(acc); },
    });
    if (signal.aborted) { disposeClassifier(c); return; }
    pretrained.current = c;
    setReady(true);
  });

  const compare = () => run(async signal => {
    const base = pretrained.current;
    if (!base) return;
    const epochs = Math.max(25, Math.min(80, Math.round(400 / shots)));
    const curves = { scratch: [], probe: [], finetune: [] } as Record<Strategy, number[]>;
    setLive(curves);
    for (const s of STRATEGIES) {
      const extractor = s.key === 'scratch' ? buildFeatureExtractor(5) : cloneExtractor(base.model.layers[0] as typeof base.model);
      const c = buildClassifier(extractor, NEW.length, s.lr, s.key === 'probe');
      try {
        await train(c, examples, tests.fresh, NEW.length, epochs, {
          batchSize: 16, random: seededRandom(6), signal,
          onEpoch: (e, acc) => {
            curves[s.key] = [...curves[s.key], acc];
            setLive({ ...curves });
            setProgress(`${s.label}: epoch ${e}/${epochs}`);
          },
        });
      } finally { disposeClassifier(c); }
      if (signal.aborted) return;
    }
    setResults(r => [...r.filter(x => x.shots !== shots), { shots, curves }].sort((a, b) => a.shots - b.shots));
  });

  const running = progress !== null;
  const final = (r: Record<Strategy, number[]>, k: Strategy) => r[k][r[k].length - 1];

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">1.</span>
        <button type="button" className={`btn btn-sm${ready ? '' : ' btn-primary'}`} onClick={pretrain} disabled={running || ready}>
          🧠 Pretrain on digits 0–4
        </button>
        {pretrainAcc !== null && <span className="dl-status">test accuracy on 0–4: {(pretrainAcc * 100).toFixed(1)}%</span>}
        <span className="demo-label tl-gap">2.</span>
        <label className="dl-slider">Examples of each new digit <strong>{shots}</strong>
          <input type="range" min={0} max={SHOTS.length - 1} value={SHOTS.indexOf(shots)} disabled={running}
            onChange={e => setShots(SHOTS[Number(e.target.value)])} />
        </label>
        <button type="button" className="btn btn-sm btn-primary" onClick={compare} disabled={running || !ready}>
          ▶ Learn 5–9 three ways
        </button>
        {running && <button type="button" className="btn btn-sm" onClick={() => abort.current?.abort()}>⏸ Stop</button>}
        {progress && <span className="dl-status">⏳ {progress}</span>}
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="dl-grid tl-grid">
        <div className="demo-panel">
          <h3>The {examples.count} training examples of 5–9</h3>
          <div className="tl-examples">
            {Array.from({ length: Math.min(examples.count, 50) }, (_, i) => (
              <PixelCanvas key={i} image={floatImage(examples.images.subarray(i * 784, (i + 1) * 784), 28, 28)} size={34} label={`A ${NEW[examples.labels[i]]}`} />
            ))}
          </div>
          {examples.count > 50 && <p className="caption">showing 50 of {examples.count}</p>}
        </div>
        <div className="demo-panel">
          <h3>Test accuracy on 5–9 while training</h3>
          <LineChart title="Test accuracy on digits 5 to 9 against epoch, for the three strategies" yMin={0} yMax={1}
            series={STRATEGIES.map(s => ({ label: s.label, color: s.color, values: live?.[s.key] ?? [] }))}
            xStart={1} xLabel="epoch" formatY={v => `${Math.round(v * 100)}%`} />
        </div>
      </div>

      {results.length > 0 && (
        <div className="demo-panel">
          <h3>Final test accuracy on 5–9</h3>
          <table className="tl-table">
            <thead><tr><th scope="col">examples per digit</th>{STRATEGIES.map(s => <th key={s.key} scope="col">{s.label}</th>)}</tr></thead>
            <tbody>
              {results.map(r => (
                <tr key={r.shots}>
                  <th scope="row">{r.shots}</th>
                  {STRATEGIES.map(s => <td key={s.key}>{(final(r.curves, s.key) * 100).toFixed(1)}%</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">Test set: all {tests.fresh.count} test images of 5–9. Try 1, 5 and 50 examples.</p>
        </div>
      )}
      {!ready && !running && <p className="dl-status tl-hint">Start by pretraining: about half a minute.</p>}
    </>
  );
}
