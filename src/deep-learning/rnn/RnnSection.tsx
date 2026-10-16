import { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import rnnCode from './rnn.ts?raw';
import pythonCode from './rnn.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { hotColor } from '../../image-processing/hough/hough';
import { DigitPicker } from '../shared/DigitPicker';
import type { MnistData } from '../shared/mnist';
import { ProbBars } from '../shared/ProbBars';
import { signedImage } from '../shared/render';
import { gatherRows, shuffled, smooth } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { accuracy, buildRecurrent, disposeRecurrent, readOne, STEPS, toSequences, trainStep, type ReadOrder, type Recurrent } from './rnn';
import '../shared/dl-shared.css';
import './rnn.css';

const BATCH = 64;
const VAL = 500;

export function RnnSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">A recurrent network reads a digit one row at a time, and has to remember what it saw</span>
        <span className="ml-auto">
          <ShowCodeButton code={rnnCode} title="rnn/rnn.ts" python={{ code: pythonCode, title: 'rnn/rnn.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>Seeing an image as a sequence</h4>
        <p>A CNN looks at local neighbourhoods everywhere at once; the Vision Transformer lets every patch look at every other
        at once. A <strong>recurrent network</strong> instead receives the image as a sequence, here the 28 rows of a digit,
        one per step, and keeps a <strong>hidden state</strong> h, a vector of 64 numbers, as its only memory. After each row a
        GRU updates it with two gates:</p>
        <p className="formula"><code>z = σ(W_z x + U_z h),  r = σ(W_r x + U_r h),  ĥ = tanh(W x + U (r ⊙ h)),  h ← (1 − z) ⊙ h + z ⊙ ĥ</code></p>
        <p>The update gate z decides how much of the memory to overwrite, the reset gate r how much of it to use when
        writing. The same weights are used at every row, so the network must learn what is worth carrying down the image.
        Gates are what let a GRU (or an LSTM) remember across many steps; a plain recurrent network's gradient fades after a
        few.</p>
        <p>Move the reading head: the probability bars show what the network would answer after reading only the rows above
        it, and the accuracy curve shows the same on 500 validation digits. Training scores only the prediction after the
        last row, so the network has no reason to commit early: most of the accuracy arrives in the last rows. Pick digits
        that look alike, such as 4 and 9 or 3 and 5, and watch when their probabilities separate. Reading by columns works
        too: it is the same network on a different sequence. Recurrent networks were the standard for text, speech and
        video before transformers.</p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [order, setOrder] = useState<ReadOrder>('rows');
  const [epochs, setEpochs] = useState(2);
  const [losses, setLosses] = useState<number[]>([]);
  const [perStep, setPerStep] = useState<number[] | null>(null);
  const [version, setVersion] = useState(0);
  const [index, setIndex] = useState(3);
  const [head, setHead] = useState(STEPS);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const model = useRef<Recurrent | null>(null);
  const abort = useRef<AbortController | null>(null);

  const dispose = () => {
    abort.current?.abort();
    if (model.current) disposeRecurrent(model.current);
    model.current = null;
  };
  useEffect(() => dispose, []);

  const reset = (nextOrder = order) => {
    dispose();
    setOrder(nextOrder);
    setLosses([]);
    setPerStep(null);
    setVersion(v => v + 1);
  };

  // A random validation sample: MNIST is stored digit by digit.
  const valRows = useMemo(() => shuffled(data.val.count, seededRandom(42)).slice(0, VAL), [data]);
  const valLabels = useMemo(() => valRows.map(r => data.val.labels[r]), [data, valRows]);
  const sequences = useMemo(() => ({
    train: toSequences(data.train.images, data.train.count, order),
    val: toSequences(gatherRows(data.val.images, 784, valRows), VAL, order),
    test: toSequences(data.test.images, data.test.count, order),
  }), [data, order, valRows]);

  const train = async () => {
    const controller = new AbortController();
    abort.current = controller;
    setError(null);
    model.current ??= buildRecurrent();
    const r = model.current;
    const random = seededRandom(losses.length + 1);
    const steps = Math.floor(data.train.count / BATCH);
    try {
      for (let e = 0; e < epochs && !controller.signal.aborted; e++) {
        const rowsOrder = shuffled(data.train.count, random);
        for (let s = 0; s < steps && !controller.signal.aborted; s++) {
          const rows = rowsOrder.slice(s * BATCH, (s + 1) * BATCH);
          const loss = trainStep(r, gatherRows(sequences.train, 784, rows), rows.map(i => data.train.labels[i]), BATCH);
          if (model.current !== r) return;
          setLosses(l => [...l, loss]);
          if (s % 5 === 0) setProgress(`epoch ${e + 1}/${epochs}, batch ${s + 1}/${steps}`);
          await tf.nextFrame();
        }
        setPerStep(Array.from({ length: STEPS }, (_, t) => accuracy(r, sequences.val, valLabels, VAL, t + 1)));
        setVersion(v => v + 1);
      }
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abort.current === controller) setProgress(null);
    }
  };

  const reading = useMemo(() => {
    const r = model.current;
    if (!r || version < 0) return null;
    return readOne(r, sequences.test.subarray(index * 784, (index + 1) * 784));
  }, [index, sequences, version]);

  const image = data.test.images.subarray(index * 784, (index + 1) * 784);
  // The digit with the part not yet read dimmed; a cyan line marks the reading head.
  const shown = useMemo(() => rgbImage(28, 28, i => {
    const x = i % 28, y = Math.floor(i / 28);
    const read = (order === 'rows' ? y : x) < head;
    const v = image[i] * 255 * (read ? 1 : 0.25);
    const atHead = (order === 'rows' ? y : x) === head - 1;
    return atHead ? [v * 0.5, 180 + v * 0.3, 220] : [v, v, v];
  }), [image, head, order]);

  const probsAtHead = reading ? reading.probs.subarray((head - 1) * 10, head * 10) : null;
  const probMap = reading ? rgbImage(STEPS, 10, i => hotColor(reading.probs[(i % STEPS) * 10 + Math.floor(i / STEPS)])) : null;
  const units = model.current?.units ?? 64;
  const hiddenMap = reading ? signedImage(Float32Array.from({ length: units * STEPS }, (_, i) => reading.hidden[(i % STEPS) * units + Math.floor(i / STEPS)]), STEPS, units, 1) : null;
  const running = progress !== null;

  return (
    <>
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={train} disabled={running}>
          {running ? `⏳ ${progress}` : losses.length ? `▶ Train ${epochs} more epoch${epochs > 1 ? 's' : ''}` : `▶ Train for ${epochs} epoch${epochs > 1 ? 's' : ''}`}
        </button>
        {running && <button type="button" className="btn btn-sm" onClick={() => abort.current?.abort()}>⏸ Stop</button>}
        <button type="button" className="btn btn-sm" onClick={() => reset()} disabled={running}>↺ Reset</button>
        <label className="dl-slider">Epochs <strong>{epochs}</strong>
          <input type="range" min={1} max={8} value={epochs} disabled={running} onChange={e => setEpochs(Number(e.target.value))} />
        </label>
        <span className="demo-label">Read by:</span>
        {(['rows', 'columns'] as const).map(o => (
          <button key={o} type="button" aria-pressed={order === o} className={`btn btn-sm${order === o ? ' btn-primary' : ''}`}
            disabled={running} onClick={() => reset(o)}>{o}</button>
        ))}
        {perStep && <span className="dl-status ml-auto">validation accuracy after all 28 {order}: {(perStep[STEPS - 1] * 100).toFixed(1)}%</span>}
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <DigitPicker split={data.test} value={index} onChange={setIndex} />

      <div className="dl-grid rnn-grid">
        <div className="demo-panel">
          <h3>Read so far: {head} {order}</h3>
          <PixelCanvas image={shown} size={196} label={`The digit, with the ${order} not yet read dimmed`} />
          <label className="dl-slider rnn-head">Reading head
            <input type="range" min={1} max={STEPS} value={head} onChange={e => setHead(Number(e.target.value))} />
          </label>
        </div>
        <div className="demo-panel">
          {probsAtHead ? <ProbBars probs={probsAtHead} trueLabel={data.test.labels[index]} title={`Prediction after ${head} ${order}`} />
            : <p className="muted">Train the network to see its predictions.</p>}
          {probMap && (
            <>
              <PixelCanvas image={probMap} size={260} label="Probability of each digit (rows of the map) after each step (columns)" />
              <p className="caption">Probability of digits 0–9 (top to bottom) after each of the 28 steps (left to right)</p>
            </>
          )}
        </div>
        <div className="demo-panel">
          <h3>Hidden state h over the steps</h3>
          {hiddenMap ? <PixelCanvas image={hiddenMap} size={260} label="The 64 hidden units (rows) after each step (columns): blue negative, red positive" />
            : <p className="muted">—</p>}
          <p className="caption">64 units (top to bottom) × 28 steps: what the network carries down the image</p>
        </div>
        <div className="demo-panel">
          <h3>How much of the image is needed</h3>
          <LineChart title={`Validation accuracy of the prediction after each number of ${order}`} yMin={0} yMax={1}
            series={[{ label: `accuracy after t ${order}`, color: 'var(--series-green)', values: perStep ?? [] }]}
            xStart={1} xLabel={`${order} read`} formatY={v => `${Math.round(v * 100)}%`} markerX={head - 1} height={170} />
          <LineChart title="Training loss, averaged over 20 batches" series={[{ label: 'loss', color: 'var(--series-red)', values: smooth(losses, 20) }]}
            xLabel="batch / 20" height={130} />
        </div>
      </div>
    </>
  );
}
