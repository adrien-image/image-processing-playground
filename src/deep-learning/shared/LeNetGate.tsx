import { useEffect, useRef, useState, type ReactNode } from 'react';
import { buildLeNet, disposeLeNet, trainModel } from '../classification/lenet';
import { publishLeNet, useSharedLeNet, type SharedLeNet } from './modelStore';
import { getMnist } from './useMnist';
import './dl-shared.css';

const QUICK_EPOCHS = 4;
/** Higher than the Classification lesson's rate, so a few epochs are enough. */
const QUICK_LEARNING_RATE = 0.002;

/**
 * Renders `children` with the shared trained LeNet. Until one exists (trained in the
 * Classification lesson, or here), offers to train one in place.
 */
export function LeNetGate({ children, purpose }: { children: (lenet: SharedLeNet) => ReactNode; purpose: string }) {
  const lenet = useSharedLeNet();
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const train = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setProgress('Loading MNIST…');
    const model = buildLeNet(QUICK_LEARNING_RATE);
    try {
      const data = await getMnist();
      await trainModel(model, data, {
        epochs: QUICK_EPOCHS,
        signal: controller.signal,
        onBatch: (b, total) => setProgress(`Training… batch ${b}/${total}`),
        onEpoch: e => {
          setProgress(`Epoch ${e.epoch}/${QUICK_EPOCHS}: validation accuracy ${(e.valAccuracy * 100).toFixed(1)}%`);
          publishLeNet(model, `a quick run here (${e.epoch} epoch${e.epoch > 1 ? 's' : ''}, ${(e.valAccuracy * 100).toFixed(1)}% validation accuracy)`);
        },
      });
      if (!controller.signal.aborted) setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
      setProgress(null);
    } finally {
      disposeLeNet(model);
    }
  };

  if (lenet && progress === null) {
    return (
      <>
        <p className="lenet-source muted">Using the LeNet trained in {lenet.source}. Train it further in the Classification lesson.</p>
        {children(lenet)}
      </>
    );
  }

  return (
    <div className="lenet-gate panel">
      <p>This lesson {purpose}, so it needs a trained digit classifier.</p>
      <p className="muted">
        Train one in the <a href="#/classification">Classification</a> lesson, or train a LeNet-5 here for {QUICK_EPOCHS} epochs
        (usually under a minute with a GPU).
      </p>
      <button type="button" className="btn btn-primary" onClick={train} disabled={progress !== null}>
        {progress === null ? `🧠 Train LeNet here (${QUICK_EPOCHS} epochs)` : 'Training…'}
      </button>
      {progress && <p className="lenet-progress mono" role="status">{progress}</p>}
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
    </div>
  );
}
