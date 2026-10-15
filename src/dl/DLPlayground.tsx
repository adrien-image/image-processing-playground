import { useCallback, useEffect, useRef, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import lenetCode from './lenet.ts?raw';
import { ShowCodeButton } from '../components/CodeViewer';
import {
  buildLeNet, disposeLeNet, evaluateTestSet, LENET_PARAMS, predictDigit, snapshotLeNet, trainModel, type EvalResult, type TrainingEpochData,
} from './lenet';
import { loadMnist, type MnistData, type MnistSplit } from './mnist';
import { toMnistInput } from './preprocess';
import { TrainingProgress } from './TrainingProgress';
import { DRAW_SIZE, DrawingCanvas } from './DrawingCanvas';
import { PredictionDisplay } from './PredictionDisplay';
import { ConfusionMatrix } from './ConfusionMatrix';
import { DigitCanvas, TestResults } from './TestResults';
import { InsideCnn, type TrainedModel } from './InsideCnn';
import './dl.css';

type Status = 'idle' | 'loading' | 'training' | 'done' | 'error';
type DlTab = 'training' | 'prediction' | 'test-results' | 'confusion-matrix' | 'inside-cnn';

const TABS: { key: DlTab; label: string; needsModel: boolean }[] = [
  { key: 'training', label: 'Training', needsModel: false },
  { key: 'prediction', label: 'Prediction', needsModel: true },
  { key: 'test-results', label: 'Test Results', needsModel: true },
  { key: 'confusion-matrix', label: 'Confusion Matrix', needsModel: true },
  { key: 'inside-cnn', label: 'Inside the CNN', needsModel: true },
];

const ARCH = [
  { name: 'Input', size: '28×28×1', color: '#6c8cff' },
  { name: 'Conv1', size: '24×24×6', color: '#a78bfa' },
  { name: 'Pool1', size: '12×12×6', color: '#f472b6' },
  { name: 'Conv2', size: '8×8×16', color: '#fb923c' },
  { name: 'Pool2', size: '4×4×16', color: '#34d399' },
  { name: 'FC', size: '256→120→84→10', color: '#f87171' },
  { name: 'Output', size: '0–9', color: '#fff' },
];

export function DLPlayground() {
  const modelRef = useRef<tf.LayersModel | null>(null);
  /** Frozen copy of the weights for "Inside the CNN", refreshed only when an epoch ends. */
  const snapshotRef = useRef<tf.LayersModel | null>(null);
  const dataRef = useRef<MnistData | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animTimers = useRef<number[]>([]);
  const epochsTrained = useRef(0);

  const [status, setStatus] = useState<Status>('idle');
  const [epochsToRun, setEpochsToRun] = useState(10);
  const [run, setRun] = useState({ epochs: 0, done: 0 });
  const [history, setHistory] = useState<TrainingEpochData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Float32Array | null>(null);
  const [networkInput, setNetworkInput] = useState<Float32Array | null>(null);
  const [archStep, setArchStep] = useState(0);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [tab, setTab] = useState<DlTab>('training');
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [emptyDrawing, setEmptyDrawing] = useState(false);
  /** The weight snapshot with a version bumped whenever it is refreshed, so views can recompute. */
  const [trained, setTrained] = useState<TrainedModel | null>(null);
  /** The held-out test split, once MNIST is loaded (state, not a ref, so the views re-render). */
  const [testSplit, setTestSplit] = useState<MnistSplit | null>(null);
  /** Last drawing that was classified, kept for the "Inside the CNN" tab. */
  const [lastDrawing, setLastDrawing] = useState<Float32Array | null>(null);

  const clearAnimation = () => {
    animTimers.current.forEach(clearTimeout);
    animTimers.current = [];
  };

  /** Free the model, its optimizer state and the snapshot (GPU memory). */
  const disposeModels = () => {
    for (const ref of [modelRef, snapshotRef]) {
      if (ref.current) disposeLeNet(ref.current);
      ref.current = null;
    }
  };

  // Leaving the section stops training and frees the model's GPU memory.
  useEffect(() => () => {
    abortRef.current?.abort();
    abortRef.current = null;
    disposeModels();
    clearAnimation();
  }, []);

  const isTrained = history.length > 0;
  const busy = status === 'loading' || status === 'training';
  // Clear the "stopping" flag once training has actually ended.
  if (stopping && !busy) setStopping(false);

  const startTraining = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    // Reset or unmount replaces/clears abortRef: this run's results are then stale.
    const stale = () => abortRef.current !== controller;
    setStatus('loading');
    setError(null);
    setPredictions(null);
    setNetworkInput(null);
    setEvalResult(null);
    setBatchProgress(null);

    try {
      if (!dataRef.current) {
        dataRef.current = await loadMnist();
        setTestSplit(dataRef.current.test);
      }
      if (stale()) return;
      if (!modelRef.current) modelRef.current = buildLeNet();
      const model = modelRef.current;
      const data = dataRef.current;
      // "Inside the CNN" looks at a copy refreshed only here, so all its views show the same
      // weights — those at the end of the last epoch — even while the next epoch trains.
      const weightsChanged = () => {
        const snapshot = snapshotLeNet(model, snapshotRef.current ?? undefined);
        snapshotRef.current = snapshot;
        setTrained(prev => ({ model: snapshot, version: (prev?.version ?? 0) + 1 }));
      };

      setStatus('training');
      setRun({ epochs: epochsToRun, done: 0 });
      await trainModel(model, data, {
        epochs: epochsToRun,
        firstEpoch: epochsTrained.current + 1,
        signal: controller.signal,
        onEpoch: (epochData) => {
          if (stale()) return;
          epochsTrained.current = epochData.epoch;
          setHistory(prev => [...prev, epochData]);
          weightsChanged();
          setRun(r => ({ ...r, done: r.done + 1 }));
          setBatchProgress(null);
        },
        onBatch: (current, total) => { if (!stale()) setBatchProgress({ current, total }); },
      });
      if (stale()) return;

      // Finished, or stopped by the user: evaluate whatever has been learned.
      setBatchProgress(null);
      if (controller.signal.aborted) weightsChanged(); // stopped mid-epoch: some batches were applied
      if (epochsTrained.current > 0) {
        setEvalResult(evaluateTestSet(model, data.test));
        setStatus('done');
      } else {
        setStatus('idle');
      }
    } catch (err) {
      if (stale()) return;
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Training failed');
    }
  };

  /** Ask training to stop after the current batch; `startTraining` then evaluates. */
  const stopTraining = () => {
    abortRef.current?.abort();
    setStopping(true);
  };

  const resetModel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    epochsTrained.current = 0;
    disposeModels();
    clearAnimation();
    setHistory([]);
    setEvalResult(null);
    setPredictions(null);
    setNetworkInput(null);
    setTrained(null);
    setLastDrawing(null);
    setRun({ epochs: 0, done: 0 });
    setBatchProgress(null);
    setStatus('idle');
    setError(null);
    setTab('training');
  };

  const classify = () => {
    const canvas = canvasRef.current;
    const model = modelRef.current;
    const ctx = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !model || !ctx) return;

    const rgba = ctx.getImageData(0, 0, DRAW_SIZE, DRAW_SIZE).data;
    const ink = new Float32Array(DRAW_SIZE * DRAW_SIZE);
    for (let i = 0; i < ink.length; i++) ink[i] = rgba[i * 4] / 255;
    const input = toMnistInput(ink, DRAW_SIZE, DRAW_SIZE);
    setNetworkInput(input);
    setEmptyDrawing(!input);
    if (!input) {
      setPredictions(null);
      return;
    }
    setLastDrawing(input);

    clearAnimation();
    setArchStep(0);
    for (let i = 1; i <= ARCH.length; i++) animTimers.current.push(window.setTimeout(() => setArchStep(i), i * 150));
    setPredictions(predictDigit(model, input));
  };

  const handleCanvasReady = useCallback((canvas: HTMLCanvasElement) => { canvasRef.current = canvas; }, []);
  const handleClear = useCallback(() => {
    setPredictions(null);
    setNetworkInput(null);
    setEmptyDrawing(false);
    setArchStep(0);
  }, []);

  const activeTab = !isTrained && TABS.find(t => t.key === tab)?.needsModel ? 'training' : tab;

  return (
    <div className="dl-playground">
      <div className="dl-topbar">
        <div className="dl-topbar-left">
          {busy ? (
            <button type="button" className="btn btn-primary" onClick={stopTraining} disabled={status === 'loading' || stopping}>
              {status === 'loading' ? '⏳ Loading MNIST…' : stopping ? '⏳ Stopping…' : `⏹ Stop (epoch ${Math.min(run.done + 1, run.epochs)}/${run.epochs})`}
            </button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={startTraining}>
              {status === 'error' ? '🔁 Retry training' : isTrained ? `🔄 Train ${epochsToRun} more epoch${epochsToRun === 1 ? '' : 's'}` : '🚀 Train LeNet on MNIST'}
            </button>
          )}
          <label className="dl-epoch-label">
            Epochs:
            <input type="number" min={1} max={20} value={epochsToRun}
              onChange={(e) => setEpochsToRun(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
              className="dl-epoch-input" disabled={busy} />
          </label>
          {isTrained && !busy && (
            <button type="button" className="btn btn-sm" onClick={resetModel}>🗑 Reset model</button>
          )}
          <ShowCodeButton code={lenetCode} title="dl/lenet.ts" open={showCode} onOpenChange={setShowCode} />
        </div>
        <div className="dl-topbar-right" role="status">
          {!isTrained && !busy && status !== 'error' && <span className="dl-model-status dl-model-untrained">⚪ Model not trained</span>}
          {isTrained && (
            <span className="dl-model-status dl-model-trained">
              ✅ Trained — {history.length} epoch{history.length > 1 ? 's' : ''}
              {evalResult && <> · Test acc: {(evalResult.accuracy * 100).toFixed(1)}%</>}
            </span>
          )}
          {error && <span className="dl-model-status dl-model-error">❌ {error}</span>}
        </div>
      </div>

      <div className="tab-bar">
        <div className="tab-list" role="tablist" aria-label="Classification views">
          {TABS.map(t => (
            <button key={t.key} type="button" role="tab" id={`dl-tab-${t.key}`} aria-controls="dl-panel"
              aria-selected={activeTab === t.key} disabled={t.needsModel && !isTrained}
              className={`tab-btn${activeTab === t.key ? ' tab-active' : ''}`} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div id="dl-panel" role="tabpanel" aria-labelledby={`dl-tab-${activeTab}`} className="dl-tab-content">
        {activeTab === 'training' && (
          <>
            {(busy || history.length > 0) && (
              <TrainingProgress history={history} runEpochs={run.epochs} runDone={run.done}
                status={status === 'loading' ? 'loading' : status === 'training' ? 'training' : 'done'}
                batchProgress={batchProgress} />
            )}
            {!busy && history.length === 0 && (
              <div className="dl-welcome">
                <h2>🧠 LeNet-5 on MNIST</h2>
                <p>Click <strong>Train LeNet on MNIST</strong> to train a convolutional neural network in your browser on 6,000 handwritten digits (600 per digit).
                After every epoch it is checked on 1,000 separate <em>validation</em> digits; a further 1,000 <em>test</em> digits are kept aside for the final evaluation.</p>
                <p>After training, explore the <strong>Prediction</strong>, <strong>Test Results</strong>, <strong>Confusion Matrix</strong> and <strong>Inside the CNN</strong> tabs.</p>
                <div className="dl-welcome-info">
                  <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">6,000</span><span className="dl-welcome-stat-label">Training</span></div>
                  <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">1,000</span><span className="dl-welcome-stat-label">Validation</span></div>
                  <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">1,000</span><span className="dl-welcome-stat-label">Test</span></div>
                  <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">{(LENET_PARAMS / 1000).toFixed(1)}k</span><span className="dl-welcome-stat-label">Parameters</span></div>
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'prediction' && (
          <div className="dl-layout">
            <div className="dl-left">
              <DrawingCanvas onCanvasReady={handleCanvasReady} onClear={handleClear} />
              <button type="button" className="btn btn-primary btn-classify" onClick={classify}>🔍 Classify</button>
              {networkInput ? (
                <div className="network-input">
                  <DigitCanvas pixels={networkInput} className="network-input-canvas" label="The 28×28 image given to the network" />
                  <p className="caption">What the network sees: your drawing scaled to a 20×20 box and centred by its centre of mass in 28×28, like the MNIST digits.</p>
                </div>
              ) : emptyDrawing && (
                <p className="note-warning" role="alert">Nothing drawn yet — draw a digit first.</p>
              )}
              <PredictionDisplay predictions={predictions} />
            </div>
            <div className="dl-right">
              <div className="arch-diagram">
                <h3>LeNet Architecture</h3>
                <div className="arch-flow">
                  {ARCH.map((layer, i) => (
                    <div key={layer.name} className="arch-step">
                      {i > 0 && <div className="arch-arrow" aria-hidden="true">▼</div>}
                      <div className={`arch-layer${archStep > i ? ' arch-layer-active' : ''}`} style={{ borderColor: layer.color }}>
                        <div className="arch-layer-name">{layer.name}</div>
                        <div className="arch-layer-size">{layer.size}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="arch-desc">
                  LeNet-5 alternates convolution and pooling, followed by fully connected layers — {LENET_PARAMS.toLocaleString()} trainable parameters
                  for 28×28 input (the original 32×32 version has ~60k).
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'test-results' && evalResult && (
          <TestResults examples={evalResult.examples} accuracy={evalResult.accuracy} total={evalResult.total} />
        )}

        {activeTab === 'confusion-matrix' && evalResult && <ConfusionMatrix matrix={evalResult.confusionMatrix} />}

        {activeTab === 'inside-cnn' && trained && (
          <InsideCnn trained={trained} drawing={lastDrawing} test={testSplit} training={busy} />
        )}
      </div>
    </div>
  );
}
