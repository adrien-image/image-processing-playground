import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useLang, useText, type Texts } from '../../lib/i18n';
import { buildLeNet, disposeLeNet, trainModel } from '../classification/lenet';
import { publishLeNet, useSharedLeNet, type SharedLeNet } from './modelStore';
import { getMnist } from './useMnist';
import './dl-shared.css';

const QUICK_EPOCHS = 4;
/** Higher than the Classification lesson's rate, so a few epochs are enough. */
const QUICK_LEARNING_RATE = 0.002;

const TEXT = {
  en: {
    loading: 'Loading MNIST…',
    batch: (b: number, total: number) => `Training… batch ${b}/${total}`,
    epoch: (e: number, acc: string) => `Epoch ${e}/${QUICK_EPOCHS}: validation accuracy ${acc}%`,
    failed: 'Training failed',
    using: (source: string) => <>Using the LeNet trained in {source}. Train it further in the Classification lesson.</>,
    needs: (purpose: string) => `This lesson ${purpose}, so it needs a trained digit classifier.`,
    offer: <>Train one in the <a href="#/classification">Classification</a> lesson, or train a LeNet-5 here for {QUICK_EPOCHS} epochs (usually under a minute with a GPU).</>,
    train: `🧠 Train LeNet here (${QUICK_EPOCHS} epochs)`,
    training: 'Training…',
  },
  fr: {
    loading: 'Chargement de MNIST…',
    batch: (b: number, total: number) => `Entraînement… lot ${b}/${total}`,
    epoch: (e: number, acc: string) => `Époque ${e}/${QUICK_EPOCHS} : précision de validation ${acc} %`,
    failed: 'L\'entraînement a échoué',
    using: (source: string) => <>Utilise le LeNet entraîné dans {source}. Poursuivez son entraînement dans la leçon Classification.</>,
    needs: (purpose: string) => `Cette leçon ${purpose}, elle a donc besoin d'un classifieur de chiffres entraîné.`,
    offer: <>Entraînez-en un dans la leçon <a href="#/classification">Classification</a>, ou entraînez ici un LeNet-5 pendant {QUICK_EPOCHS} époques (en général moins d'une minute avec un GPU).</>,
    train: `🧠 Entraîner LeNet ici (${QUICK_EPOCHS} époques)`,
    training: 'Entraînement…',
  },
};

/** Where a quick run's weights came from, in both languages. */
const quickSource = (epochs: number, acc: string): Texts<string> => ({
  en: `a quick run here (${epochs} epoch${epochs > 1 ? 's' : ''}, ${acc}% validation accuracy)`,
  fr: `un entraînement rapide ici (${epochs} époque${epochs > 1 ? 's' : ''}, ${acc} % de précision de validation)`,
});

/**
 * Renders `children` with the shared trained LeNet. Until one exists (trained in the
 * Classification lesson, or here), offers to train one in place.
 */
export function LeNetGate({ children, purpose }: { children: (lenet: SharedLeNet) => ReactNode; purpose: string }) {
  const lenet = useSharedLeNet();
  const lang = useLang();
  const t = useText(TEXT);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const train = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setProgress(t.loading);
    const model = buildLeNet(QUICK_LEARNING_RATE);
    try {
      const data = await getMnist();
      await trainModel(model, data, {
        epochs: QUICK_EPOCHS,
        signal: controller.signal,
        onBatch: (b, total) => setProgress(t.batch(b, total)),
        onEpoch: e => {
          const acc = (e.valAccuracy * 100).toFixed(1);
          setProgress(t.epoch(e.epoch, acc));
          publishLeNet(model, quickSource(e.epoch, acc));
        },
      });
      if (!controller.signal.aborted) setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.failed);
      setProgress(null);
    } finally {
      disposeLeNet(model);
    }
  };

  if (lenet && progress === null) {
    return (
      <>
        <p className="lenet-source muted">{t.using(lenet.source[lang])}</p>
        {children(lenet)}
      </>
    );
  }

  return (
    <div className="lenet-gate panel">
      <p>{t.needs(purpose)}</p>
      <p className="muted">{t.offer}</p>
      <button type="button" className="btn btn-primary" onClick={train} disabled={progress !== null}>
        {progress === null ? t.train : t.training}
      </button>
      {progress && <p className="lenet-progress mono" role="status">{progress}</p>}
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
    </div>
  );
}
