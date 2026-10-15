import { useEffect, useRef, useState } from 'react';
import regCode from './regularise.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { CLASS_COLORS } from '../shared/render';
import { useMnist } from '../shared/useMnist';
import { balancedSubset, buildMlp, trainRun, type EpochStats, type RegConfig } from './regularise';
import '../shared/dl-shared.css';
import './regularisation.css';

const SIZES = [100, 300, 1000, 6000];
const DROPOUTS = [0, 0.25, 0.5];
const L2S = [0, 1e-4, 1e-3];

interface Run {
  id: number;
  config: RegConfig;
  history: EpochStats[];
  done: boolean;
}

const describe = (c: RegConfig) =>
  [`${c.trainSize} images`, c.dropout ? `dropout ${c.dropout}` : null, c.l2 ? `L2 ${c.l2}` : null, c.augment ? 'augmented' : null]
    .filter(Boolean).join(' · ') + (c.dropout || c.l2 || c.augment ? '' : ' · no regularisation');

export function RegularisationSection() {
  const { data, error } = useMnist();
  const [config, setConfig] = useState<RegConfig>({ trainSize: 300, dropout: 0, l2: 0, augment: false });
  const [epochs, setEpochs] = useState(30);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [showCode, setShowCode] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const running = runs.some(r => !r.done);

  useEffect(() => () => abortRef.current?.abort(), []);

  const start = async () => {
    if (!data) return;
    const controller = new AbortController();
    abortRef.current = controller;
    const id = (runs[runs.length - 1]?.id ?? 0) + 1;
    const runConfig = { ...config };
    setRuns(rs => [...rs.slice(-7), { id, config: runConfig, history: [], done: false }]);
    setSelected(id);
    const model = buildMlp(runConfig);
    try {
      await trainRun(model, balancedSubset(data.train, runConfig.trainSize), data.val, runConfig.augment, seededRandom(id), {
        epochs,
        signal: controller.signal,
        onEpoch: s => setRuns(rs => rs.map(r => (r.id === id ? { ...r, history: [...r.history, s] } : r))),
      });
    } finally {
      model.dispose();
      setRuns(rs => rs.map(r => (r.id === id ? { ...r, done: true } : r)));
    }
  };

  const shown = runs.find(r => r.id === selected) ?? runs[runs.length - 1];
  const set = <K extends keyof RegConfig>(key: K, value: RegConfig[K]) => setConfig(c => ({ ...c, [key]: value }));
  const chips = <K extends 'trainSize' | 'dropout' | 'l2'>(key: K, values: number[], format: (v: number) => string) =>
    values.map(v => (
      <button key={v} type="button" aria-pressed={config[key] === v} disabled={running}
        className={`btn btn-sm${config[key] === v ? ' btn-primary' : ''}`} onClick={() => set(key, v)}>{format(v)}</button>
    ));

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Training images:</span>{chips('trainSize', SIZES, v => String(v))}
        <span className="demo-label reg-gap">Dropout:</span>{chips('dropout', DROPOUTS, v => (v ? String(v) : 'off'))}
        <span className="demo-label reg-gap">L2:</span>{chips('l2', L2S, v => (v ? v.toExponential(0) : 'off'))}
        <label className="dl-slider reg-gap">
          <input type="checkbox" checked={config.augment} disabled={running} onChange={e => set('augment', e.target.checked)} /> Augmentation
        </label>
        <span className="ml-auto">
          <ShowCodeButton code={regCode} title="regularisation/regularise.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <label className="dl-slider">Epochs <strong>{epochs}</strong>
          <input type="range" min={5} max={60} step={5} value={epochs} disabled={running} onChange={e => setEpochs(Number(e.target.value))} />
        </label>
        {running ? (
          <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>⏹ Stop</button>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" onClick={start} disabled={!data}>
            {data ? '▶ Train this configuration' : 'Loading MNIST…'}
          </button>
        )}
        {runs.length > 0 && !running && <button type="button" className="btn btn-sm" onClick={() => { setRuns([]); setSelected(null); }}>Clear runs</button>}
        <span className="muted reg-note">A 784-256-256-10 perceptron; validation on 1,000 held-out digits.</span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {shown && (
        <div className="dl-grid reg-charts">
          <div className="demo-panel">
            <h3>Loss: {describe(shown.config)}</h3>
            <LineChart
              title="Training and validation loss per epoch"
              series={[
                { label: 'training', color: '#60a5fa', values: shown.history.map(h => h.trainLoss), dashed: true },
                { label: 'validation', color: '#f87171', values: shown.history.map(h => h.valLoss) },
              ]}
              xStart={1}
              xLabel="epoch"
              yMin={0}
            />
          </div>
          <div className="demo-panel">
            <h3>Accuracy</h3>
            <LineChart
              title="Training and validation accuracy per epoch"
              series={[
                { label: 'training', color: '#60a5fa', values: shown.history.map(h => h.trainAccuracy), dashed: true },
                { label: 'validation', color: '#34d399', values: shown.history.map(h => h.valAccuracy) },
              ]}
              xStart={1}
              xLabel="epoch"
              formatY={v => `${Math.round(v * 100)}%`}
            />
          </div>
        </div>
      )}

      {runs.length > 0 && (
        <div className="demo-panel reg-runs">
          <h3>Runs</h3>
          <table>
            <thead>
              <tr><th>Configuration</th><th>Train acc.</th><th>Val. acc.</th><th>Gap</th><th>Best val.</th></tr>
            </thead>
            <tbody>
              {runs.map((r, k) => {
                const last = r.history[r.history.length - 1];
                const best = r.history.reduce((m, h) => Math.max(m, h.valAccuracy), 0);
                const gap = last ? last.trainAccuracy - last.valAccuracy : 0;
                return (
                  <tr key={r.id} className={r.id === shown?.id ? 'reg-selected' : undefined}>
                    <td>
                      <button type="button" className="reg-run-btn" onClick={() => setSelected(r.id)}>
                        <span className="reg-swatch" style={{ background: CLASS_COLORS[k % 10] }} />
                        {describe(r.config)}{r.done ? '' : ` (epoch ${r.history.length}/${epochs})`}
                      </button>
                    </td>
                    <td className="mono">{last ? pct(last.trainAccuracy) : '—'}</td>
                    <td className="mono">{last ? pct(last.valAccuracy) : '—'}</td>
                    <td className={`mono ${gap > 0.1 ? 'text-error' : gap > 0.04 ? '' : 'text-success'}`}>{last ? pct(gap) : '—'}</td>
                    <td className="mono">{last ? pct(best) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {runs.length > 1 && (
            <LineChart
              title="Validation accuracy of every run"
              series={runs.map((r, k) => ({ label: `run ${k + 1}`, color: CLASS_COLORS[k % 10], values: r.history.map(h => h.valAccuracy) }))}
              xStart={1}
              xLabel="epoch"
              formatY={v => `${Math.round(v * 100)}%`}
              height={180}
            />
          )}
        </div>
      )}

      <div className="demo-explanation">
        <h4>Memorising vs. generalising</h4>
        <p>Train on 100 or 300 images with no regularisation: training accuracy reaches 100% within a few epochs while
        validation accuracy stalls far below, and the validation <em>loss</em> stops falling and creeps back up. The network
        has memorised its training set. The <strong>gap</strong> between the two curves is overfitting.</p>
        <p><strong>Dropout</strong> switches off a random half (or quarter) of the hidden units at every step, so no unit can rely
        on another and the network learns redundant features. <strong>L2 weight decay</strong> adds λ·Σw² to the loss and keeps
        weights small, which makes the function smoother. <strong>Augmentation</strong> shows a randomly shifted and rotated copy
        of each digit every epoch, so the network can't memorise exact pixels (see the Augmentation lesson).</p>
        <p>Each remedy alone narrows the gap only a little on 300 images; combine them, then try 6,000 images with nothing
        switched on. More data usually beats every trick.</p>
      </div>
    </div>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
