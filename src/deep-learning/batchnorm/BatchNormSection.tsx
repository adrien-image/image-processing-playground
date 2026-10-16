import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import batchnormCode from './batchnorm.ts?raw';
import pythonCode from './batchnorm.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import type { MnistData } from '../shared/mnist';
import { gatherRows, shuffled } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { accuracy, buildMlp, disposeModel, percentiles, probeModel, trainBatch, unitInputs } from './batchnorm';
import '../shared/dl-shared.css';
import './batchnorm.css';

const BATCH = 60;
const EVERY = 20;
const PROBE_IMAGES = 500;
const QUANTILES = [15, 50, 85];

interface Point { step: number; accPlain: number; accBn: number; plain: number[]; bn: number[] }

export function BatchNormSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">The same sigmoid network, trained on the same batches, with and without batch normalisation</span>
        <span className="ml-auto">
          <ShowCodeButton code={batchnormCode} title="batchnorm/batchnorm.ts" python={{ code: pythonCode, title: 'batchnorm/batchnorm.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>Keep every layer's inputs in range</h4>
        <p>When a network trains, every layer's weights change, so the inputs of each layer keep shifting under it: the layer
        has to chase a moving target. With sigmoid units it is worse: once a unit's input drifts to ±5 the sigmoid is flat
        there and its gradient vanishes. <strong>Batch normalisation</strong> standardises each unit over the mini-batch and
        then lets the network choose its scale and offset:</p>
        <p className="formula"><code>x̂ = (x − μ_B) / √(σ²_B + ε),   y = γ · x̂ + β</code></p>
        <p>The curves on the right follow Figure 1 of the original paper: the 15th, 50th and 85th percentiles of one unit's
        input to its sigmoid, over training. Without batch norm that distribution is the product of every earlier layer, so it
        shifts whenever they change. With it, the distribution is set only by the unit's own γ and β, which change slowly and
        on purpose. In this small network both widen as the digits get separated; the clearest difference is the accuracy,
        which the normalised network reaches much sooner. Raise the learning rate to 2 or more and the gap grows: the plain
        network stalls with saturated units while the normalised one keeps learning.</p>
        <p>At test time there is no batch to average over, so the layer uses <strong>running averages</strong> of μ and σ²
        collected during training (that is what the percentile curves use). The β parameter replaces the dense layer's bias,
        which the subtraction of μ would cancel anyway. Batch norm is in almost every CNN since 2015, though transformers
        prefer layer normalisation, which averages over the units of one example instead of over the batch.</p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [logLr, setLogLr] = useState(Math.log10(0.5));
  const [stepsToRun, setStepsToRun] = useState(600);
  const [unit, setUnit] = useState(7);
  const [history, setHistory] = useState<Point[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const models = useRef<{ plain: tf.LayersModel; bn: tf.LayersModel; lr: number } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const stepRef = useRef(0);
  const orderRef = useRef<number[]>([]);
  const lr = 10 ** logLr;

  const dispose = () => {
    abort.current?.abort();
    if (models.current) { disposeModel(models.current.plain); disposeModel(models.current.bn); }
    models.current = null;
  };
  useEffect(() => dispose, []);

  const reset = () => {
    dispose();
    stepRef.current = 0;
    setHistory([]);
    setProgress(null);
  };

  const train = async () => {
    if (models.current && models.current.lr !== lr) reset();
    const controller = new AbortController();
    abort.current = controller;
    setError(null);
    models.current ??= { plain: buildMlp({ batchNorm: false, learningRate: lr }), bn: buildMlp({ batchNorm: true, learningRate: lr }), lr };
    const { plain, bn } = models.current;
    const probes = { plain: probeModel(plain), bn: probeModel(bn) };
    // A random sample: MNIST is stored digit by digit.
    const probeImages = gatherRows(data.val.images, 784, shuffled(data.val.count, seededRandom(42)).slice(0, PROBE_IMAGES));
    const random = seededRandom(stepRef.current + 1);
    const perEpoch = Math.floor(data.train.count / BATCH);
    const record = () => {
      const q = (m: tf.LayersModel) => percentiles(unitInputs(m, probeImages, PROBE_IMAGES, unit), QUANTILES);
      return {
        step: stepRef.current,
        accPlain: accuracy(plain, data.val.images, data.val.labels, data.val.count),
        accBn: accuracy(bn, data.val.images, data.val.labels, data.val.count),
        plain: q(probes.plain), bn: q(probes.bn),
      };
    };
    try {
      if (stepRef.current === 0) setHistory([record()]);
      for (let s = 0; s < stepsToRun && !controller.signal.aborted; s++) {
        // A new random order at the start of every epoch.
        const b = stepRef.current % perEpoch;
        if (b === 0 || orderRef.current.length === 0) orderRef.current = shuffled(data.train.count, random);
        const rows = orderRef.current.slice(b * BATCH, (b + 1) * BATCH);
        const images = gatherRows(data.train.images, 784, rows);
        const labels = rows.map(r => data.train.labels[r]);
        await trainBatch(plain, images, labels, BATCH);
        await trainBatch(bn, images, labels, BATCH);
        stepRef.current++;
        if (stepRef.current % EVERY === 0) {
          if (controller.signal.aborted || models.current?.plain !== plain) break;
          const point = record();
          setHistory(h => [...h, point]);
          setProgress(`step ${s + 1}/${stepsToRun}`);
          await tf.nextFrame();
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abort.current === controller) setProgress(null);
    }
  };

  const running = progress !== null;
  const latest = history[history.length - 1];
  const yRange = history.length > 1 ? (() => {
    const all = history.flatMap(h => [...h.plain, ...h.bn]);
    return [Math.min(...all), Math.max(...all)];
  })() : [-1, 1];
  const quantileSeries = (key: 'plain' | 'bn') => QUANTILES.map((q, i) => ({
    label: `${q}th percentile`, color: ['var(--series-blue)', 'var(--series-green)', 'var(--series-orange)'][i], values: history.map(h => h[key][i]), dashed: q !== 50,
  }));

  return (
    <>
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={train} disabled={running}>
          {running ? `⏳ Training… ${progress}` : history.length ? `▶ Train ${stepsToRun} more steps` : `▶ Train both for ${stepsToRun} steps`}
        </button>
        {running && <button type="button" className="btn btn-sm" onClick={() => abort.current?.abort()}>⏸ Stop</button>}
        <button type="button" className="btn btn-sm" onClick={reset} disabled={running}>↺ Reset</button>
        <div className="dl-sliders">
          <label className="dl-slider">Learning rate <strong>{lr.toFixed(lr < 1 ? 2 : 1)}</strong>
            <input type="range" min={-1.5} max={0.8} step={0.05} value={logLr} disabled={running} onChange={e => setLogLr(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Steps per run <strong>{stepsToRun}</strong>
            <input type="range" min={100} max={2000} step={100} value={stepsToRun} disabled={running} onChange={e => setStepsToRun(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Watched unit <strong>#{unit}</strong>
            <input type="range" min={0} max={99} value={unit} disabled={running || history.length > 0} onChange={e => setUnit(Number(e.target.value))} />
          </label>
        </div>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {models.current && models.current.lr !== lr && <p className="dl-status">A new learning rate restarts both networks from the same initial weights.</p>}

      {latest && (
        <div className="dl-stat-row">
          <span>step <strong>{latest.step}</strong></span>
          <span>validation accuracy without BN <strong>{(latest.accPlain * 100).toFixed(1)}%</strong></span>
          <span>with BN <strong>{(latest.accBn * 100).toFixed(1)}%</strong></span>
        </div>
      )}

      <div className="dl-grid bn-grid">
        <div className="demo-panel">
          <h3>Validation accuracy</h3>
          <LineChart title="Validation accuracy against training step, with and without batch normalisation" yMin={0} yMax={1}
            series={[
              { label: 'without BN', color: 'var(--series-red)', values: history.map(h => h.accPlain) },
              { label: 'with BN', color: 'var(--series-green)', values: history.map(h => h.accBn) },
            ]}
            xTickLabel={i => String(history[i]?.step ?? i * EVERY)} xLabel="step" formatY={v => `${Math.round(v * 100)}%`} />
        </div>
        <div className="demo-panel">
          <h3>Unit #{unit}'s sigmoid input · without BN</h3>
          <LineChart title={`Percentiles of unit ${unit}'s input to its sigmoid, without batch normalisation`} yMin={yRange[0]} yMax={yRange[1]}
            series={quantileSeries('plain')} xTickLabel={i => String(history[i]?.step ?? i * EVERY)} xLabel="step" />
        </div>
        <div className="demo-panel">
          <h3>Unit #{unit}'s sigmoid input · with BN</h3>
          <LineChart title={`Percentiles of unit ${unit}'s input to its sigmoid, with batch normalisation`} yMin={yRange[0]} yMax={yRange[1]}
            series={quantileSeries('bn')} xTickLabel={i => String(history[i]?.step ?? i * EVERY)} xLabel="step" />
        </div>
      </div>
      {!history.length && <p className="dl-status bn-hint">Press Train: both networks start from the same weights and see the same batches.</p>}
    </>
  );
}
