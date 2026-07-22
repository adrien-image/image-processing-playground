import { CodeViewer } from '../components/CodeViewer';
import lenetCode from './lenet.ts?raw';
import { useState, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { buildLeNet, loadMnistData, trainModel, predictDigit, evaluateTestSet, type TrainingEpochData, type EvalResult } from './lenet';
import { TrainingProgress } from './TrainingProgress';
import { DrawingCanvas } from './DrawingCanvas';
import { PredictionDisplay } from './PredictionDisplay';
import { ConfusionMatrix } from './ConfusionMatrix';
import { TestResults } from './TestResults';

type Status = 'idle' | 'loading' | 'training' | 'done' | 'error' | 'predicting';
type DlTab = 'training' | 'prediction' | 'test-results' | 'confusion-matrix';

export function DLPlayground() {
  const modelRef = useRef<tf.LayersModel | null>(null);
  const dataRef = useRef<Awaited<ReturnType<typeof loadMnistData>> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [status, setStatus] = useState<Status>('idle');
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [totalEpochs, setTotalEpochs] = useState(10);
  const [history, setHistory] = useState<TrainingEpochData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Float32Array | null>(null);
  const [archStep, setArchStep] = useState(0);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [dlTab, setDlTab] = useState<DlTab>('training');
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);

  const isTrained = modelRef.current !== null && history.length > 0;
  const isTraining = status === 'training' || status === 'loading';
  const trainedTabs = ['prediction', 'test-results', 'confusion-matrix'];

  const startTraining = useCallback(async () => {
    setStatus('loading');
    setError(null);
    if (!isTrained) setHistory([]);
    setPredictions(null);
    setEvalResult(null);

    try {
      if (!dataRef.current) dataRef.current = await loadMnistData();
      if (!modelRef.current) modelRef.current = buildLeNet();

      setStatus('training');
      await trainModel(modelRef.current, dataRef.current, totalEpochs, 128, (data) => {
        setHistory(prev => [...prev, data]);
        setCurrentEpoch(data.epoch);
        setBatchProgress(null);
      }, (_batch, _total) => {
        setBatchProgress({ current: _batch, total: _total });
      });

      // Evaluate on test set
      const result = evaluateTestSet(modelRef.current, dataRef.current);
      setEvalResult(result);
      setStatus('done');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Training failed');
    }
  }, [totalEpochs, isTrained]);

  const resetModel = useCallback(() => {
    modelRef.current = null;
    setHistory([]);
    setEvalResult(null);
    setPredictions(null);
    setCurrentEpoch(0);
    setBatchProgress(null);
    setStatus('idle');
    setError(null);
  }, []);

  const classify = useCallback(async () => {
    const canvas = canvasRef.current;
    const model = modelRef.current;
    if (!canvas || !model) return;
    setStatus('predicting');
    setPredictions(null);
    setArchStep(0);

    // Clear any existing animation timers
    if (animTimers.current.length) {
      animTimers.current.forEach(clearTimeout);
      animTimers.current = [];
    }

    // Start architecture animation independently (non-blocking)
    for (let i = 1; i <= 6; i++) {
      animTimers.current.push(setTimeout(() => setArchStep(i), i * 150));
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28; tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) pixels[i] = imageData.data[i * 4] / 255;
    const tensor = tf.tensor4d(pixels, [1, 28, 28, 1]);
    const probs = predictDigit(model, tensor);
    tensor.dispose();
    setPredictions(probs);
    setTimeout(() => setArchStep(7), 1050);
    setStatus('done');
  }, []);

  const handleCanvasReady = useCallback((canvas: HTMLCanvasElement) => { canvasRef.current = canvas; }, []);
  const handleClear = useCallback(() => { setPredictions(null); setArchStep(0); }, []);

  // Redirect to training tab if not trained and trying to access trained-only tabs
  const activeTab = !isTrained && trainedTabs.includes(dlTab) ? 'training' : dlTab;

  return (
    <>
    <div className="dl-playground">
      {/* Top bar */}
      <div className="dl-topbar">
        <div className="dl-topbar-left">
          <button className="btn btn-primary" onClick={startTraining} disabled={isTraining}>
            {status === 'idle' && '🚀 Train LeNet on MNIST'}
            {status === 'done' && '🔄 Train more epochs'}
            {status === 'loading' && '⏳ Loading MNIST data…'}
            {status === 'training' && `⏳ Epoch ${currentEpoch}/${totalEpochs}`}
          </button>
          <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
          {isTrained && (
            <button className="btn btn-sm" onClick={resetModel} disabled={isTraining} style={{ marginLeft: '0.5rem' }}>
              🔄 Reset model
            </button>
          )}
          <label className="dl-epoch-label">
            Epochs:
            <input type="number" min={1} max={20} value={totalEpochs}
              onChange={(e) => setTotalEpochs(Math.max(1, Math.min(20, Number(e.target.value))))}
              className="dl-epoch-input" disabled={isTraining} />
          </label>
        </div>
        <div className="dl-topbar-right">
          {status === 'idle' && !isTrained && <span className="dl-model-status dl-model-untrained">⚪ Model not trained</span>}
          {isTrained && (
            <span className="dl-model-status dl-model-trained">
              ✅ Trained — {history.length} epoch{history.length > 1 ? 's' : ''}
              {evalResult && <> · Test acc: {(evalResult.accuracy * 100).toFixed(1)}%</>}
            </span>
          )}
          {error && <span className="dl-model-status dl-model-error">❌ {error}</span>}
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="tab-bar">
        <button className={`tab-btn${activeTab === 'training' ? ' tab-active' : ''}`} onClick={() => setDlTab('training')}>Training</button>
        <button className={`tab-btn${activeTab === 'prediction' ? ' tab-active' : ''}`} onClick={() => setDlTab('prediction')} disabled={!isTrained}>Prediction</button>
        <button className={`tab-btn${activeTab === 'test-results' ? ' tab-active' : ''}`} onClick={() => setDlTab('test-results')} disabled={!isTrained}>Test Results</button>
        <button className={`tab-btn${activeTab === 'confusion-matrix' ? ' tab-active' : ''}`} onClick={() => setDlTab('confusion-matrix')} disabled={!isTrained}>Confusion Matrix</button>
      </div>

      {/* Training tab */}
      {activeTab === 'training' && (
        <div className="dl-tab-content">
          {isTraining && <TrainingProgress history={history} currentEpoch={currentEpoch} totalEpochs={totalEpochs} status={status === 'loading' ? 'loading' : 'training'} batchProgress={batchProgress} />}
          {status === 'idle' && !isTrained && (
            <div className="dl-welcome">
              <h2>🧠 LeNet-5 on MNIST</h2>
              <p>Click <strong>Train LeNet on MNIST</strong> above to start training a convolutional neural network on 6,000 handwritten digit images (600 per digit).</p>
              <p>The model learns to recognise digits 0–9. After training, explore the <strong>Prediction</strong>, <strong>Test Results</strong>, and <strong>Confusion Matrix</strong> tabs.</p>
              <div className="dl-welcome-info">
                <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">6,000</span><span className="dl-welcome-stat-label">Training samples</span></div>
                <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">1,000</span><span className="dl-welcome-stat-label">Test samples</span></div>
                <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">60k</span><span className="dl-welcome-stat-label">Parameters</span></div>
                <div className="dl-welcome-stat"><span className="dl-welcome-stat-value">{totalEpochs}</span><span className="dl-welcome-stat-label">Epochs</span></div>
              </div>
            </div>
          )}
          {isTrained && status !== 'training' && <TrainingProgress history={history} currentEpoch={history.length} totalEpochs={history.length} status="done" />}
        </div>
      )}

      {/* Prediction tab */}
      {activeTab === 'prediction' && (
        <div className="dl-tab-content">
          <div className="dl-layout">
            <div className="dl-left">
              <DrawingCanvas onCanvasReady={handleCanvasReady} onClear={handleClear} />
              <button className="btn btn-primary btn-classify" onClick={classify}>🔍 Classify</button>
              <PredictionDisplay predictions={predictions} />
            </div>
            <div className="dl-right">
              <div className="arch-diagram">
                <h3>LeNet Architecture</h3>
                <div className="arch-flow">
                  <ArchLayer name="Input" size="28×28×1" color="#6c8cff" active={archStep >= 1} />
                  <ArchArrow />
                  <ArchLayer name="Conv1" size="24×24×6" color="#a78bfa" active={archStep >= 2} />
                  <ArchArrow />
                  <ArchLayer name="Pool1" size="12×12×6" color="#f472b6" active={archStep >= 3} />
                  <ArchArrow />
                  <ArchLayer name="Conv2" size="8×8×16" color="#fb923c" active={archStep >= 4} />
                  <ArchArrow />
                  <ArchLayer name="Pool2" size="4×4×16" color="#34d399" active={archStep >= 5} />
                  <ArchArrow />
                  <ArchLayer name="FC" size="120→84→10" color="#f87171" active={archStep >= 6} />
                  <ArchArrow />
                  <ArchLayer name="Output" size="0–9" color="#fff" active={archStep >= 7} />
                </div>
                <p className="arch-desc">LeNet-5 alternates convolution and pooling, followed by fully connected layers. ~60k parameters.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Results tab */}
      {activeTab === 'test-results' && evalResult && (
        <div className="dl-tab-content">
          <TestResults examples={evalResult.examples} accuracy={evalResult.accuracy} />
        </div>
      )}

      {/* Confusion Matrix tab */}
      {activeTab === 'confusion-matrix' && evalResult && (
        <div className="dl-tab-content">
          <ConfusionMatrix matrix={evalResult.confusionMatrix} />
        </div>
      )}
    </div>
      {showCode && <CodeViewer code={lenetCode} title="lenet.ts" onClose={() => setShowCode(false)} />}
    </>
  );
}

function ArchLayer({ name, size, color, active }: { name: string; size: string; color: string; active: boolean }) {
  return (
    <>
    <div className={`arch-layer${active ? ' arch-layer-active' : ''}`} style={{ borderColor: color }}>
      <div className="arch-layer-name">{name}</div>
      <div className="arch-layer-size">{size}</div>
    </div>
    </>
  );
}

function ArchArrow() {
  return <div className="arch-arrow">▼</div>;
}
