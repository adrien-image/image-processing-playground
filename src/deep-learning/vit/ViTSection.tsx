import { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import vitCode from './vit.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { hotColor } from '../../image-processing/hough/hough';
import { rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { DigitPicker } from '../shared/DigitPicker';
import type { MnistData } from '../shared/mnist';
import { ProbBars } from '../shared/ProbBars';
import { divergingColor } from '../../lib/colormap';
import { gatherRows, optimise, shuffled, smooth } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { attentionRollout, GRID, PATCH, PATCHES, patchify, TinyViT, TOKENS } from './vit';
import '../shared/dl-shared.css';
import './vit.css';

const BATCH = 64;
const DISPLAY = 252;

type View = { kind: 'head'; layer: number; head: number } | { kind: 'mean'; layer: number } | { kind: 'rollout' };

export function ViTSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">A transformer that reads a digit as 16 patches and lets every patch attend to every other</span>
        <span className="ml-auto">
          <ShowCodeButton code={vitCode} title="vit/vit.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>Attention instead of convolution</h4>
        <p>A Vision Transformer cuts the image into patches (here 16 patches of 7×7 pixels), turns each into a vector
        ("token") with one linear layer, prepends a learned <strong>[CLS]</strong> token and adds a learned
        <strong> position embedding</strong> to each. Transformer blocks then let every token update itself with a weighted
        average of all the others, the weights coming from <strong>self-attention</strong>:</p>
        <p className="formula"><code>Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V</code></p>
        <p>The prediction is read from the [CLS] token, so the attention maps show which patches it gathered information from.
        Click a patch to see what <em>that</em> patch attends to instead. Each head learns its own pattern. Rollout combines
        all blocks and heads.</p>
        <p>A convolution only ever looks at neighbours; a transformer starts with no idea which patches are near each other.
        The position-similarity grid shows it learning the layout: after training, each patch's embedding resembles its
        neighbours'. That missing built-in prior is why ViTs need more data than CNNs: compare this model's accuracy with
        LeNet's in the Classification lesson.</p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [epochsToRun, setEpochsToRun] = useState(4);
  const [losses, setLosses] = useState<number[]>([]);
  const [accuracy, setAccuracy] = useState<number[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [index, setIndex] = useState(700);
  const [view, setView] = useState<View>({ kind: 'rollout' });
  const [query, setQuery] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<TinyViT | null>(null);
  const optRef = useRef<tf.Optimizer | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const dispose = () => {
    abortRef.current?.abort();
    modelRef.current?.dispose();
    optRef.current?.dispose();
    modelRef.current = null;
    optRef.current = null;
  };
  useEffect(() => dispose, []);

  const reset = () => {
    dispose();
    setLosses([]);
    setAccuracy([]);
    setVersion(v => v + 1);
  };

  const train = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    modelRef.current ??= new TinyViT();
    optRef.current ??= tf.train.adam(0.002);
    const model = modelRef.current, optimizer = optRef.current;
    const random = seededRandom(accuracy.length + 3);
    const steps = Math.floor(data.train.count / BATCH);
    const patches = patchify(data.train.images, data.train.count);
    try {
      for (let e = 0; e < epochsToRun && !controller.signal.aborted; e++) {
        const order = shuffled(data.train.count, random);
        const stepLosses = await optimise(optimizer, steps, s => {
          const rows = order.slice(s * BATCH, (s + 1) * BATCH);
          const x = tf.tensor3d(gatherRows(patches, PATCHES * PATCH * PATCH, rows), [BATCH, PATCHES, PATCH * PATCH]);
          const y = tf.oneHot(tf.tensor1d(rows.map(r => data.train.labels[r]), 'int32'), 10) as tf.Tensor2D;
          return model.loss(x, y);
        }, {
          signal: controller.signal,
          varList: model.variables,
          onStep: s => { if (s % 10 === 0) setProgress(`epoch ${e + 1}/${epochsToRun}, batch ${s + 1}/${steps}`); },
        });
        if (controller.signal.aborted || modelRef.current !== model) break;
        setLosses(l => [...l, ...stepLosses]);
        setAccuracy(a => [...a, model.accuracy(data.val.images, data.val.labels, data.val.count)]);
        setVersion(v => v + 1);
      }
    } catch (err) {
      // Reset or leaving the page disposes the model mid-training: that is a normal stop.
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abortRef.current === controller) setProgress(null);
    }
  };

  const image = data.test.images.subarray(index * 784, (index + 1) * 784);
  const result = useMemo(() => {
    const model = modelRef.current;
    if (!model || version < 0) return null;
    const { probs, attention } = model.run(image, 1);
    return { probs, attention, similarity: model.positionSimilarity(), heads: model.config.heads, layers: model.config.layers };
  }, [image, version]);

  /** The 16 patch weights to draw, for the selected view and query token. */
  const weights = useMemo(() => {
    if (!result) return null;
    const { attention, heads } = result;
    const row = query === null ? 0 : query + 1; // token 0 is [CLS]
    if (view.kind === 'rollout') {
      if (query === null) return attentionRollout(attention, heads);
    }
    const layer = view.kind === 'rollout' ? result.layers - 1 : view.layer;
    const a = attention[layer];
    const out = new Float32Array(PATCHES);
    const hs = view.kind === 'head' ? [view.head] : Array.from({ length: heads }, (_, h) => h);
    for (const h of hs) for (let p = 0; p < PATCHES; p++) out[p] += a[(h * TOKENS + row) * TOKENS + p + 1] / hs.length;
    return out;
  }, [result, view, query]);

  const overlay = useMemo(() => {
    if (!weights) return null;
    const max = Math.max(...weights) || 1;
    return rgbImage(28, 28, i => {
      const x = i % 28, y = Math.floor(i / 28);
      const p = Math.floor(y / PATCH) * GRID + Math.floor(x / PATCH);
      const [r, g, b] = hotColor((weights[p] / max) * 0.95);
      const ink = image[i] * 0.55;
      return [r + (255 - r) * ink, g + (255 - g) * ink, b + (255 - b) * ink];
    });
  }, [weights, image]);

  const views: { label: string; v: View }[] = result
    ? [
        { label: 'Rollout (all blocks)', v: { kind: 'rollout' } },
        ...Array.from({ length: result.layers }, (_, l) => [
          { label: `Block ${l + 1} · mean`, v: { kind: 'mean', layer: l } as View },
          ...Array.from({ length: result.heads }, (_, h) => ({ label: `Block ${l + 1} · head ${h + 1}`, v: { kind: 'head', layer: l, head: h } as View })),
        ]).flat(),
      ]
    : [];
  const same = (a: View, b: View) => JSON.stringify(a) === JSON.stringify(b);

  return (
    <>
      <div className="demo-controls">
        {progress ? (
          <>
            <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>⏹ Stop</button>
            <span className="dl-status">{progress}</span>
          </>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" onClick={train}>▶ Train {accuracy.length ? 'more' : ''} ({epochsToRun} epochs)</button>
        )}
        <label className="dl-slider">Epochs <strong>{epochsToRun}</strong>
          <input type="range" min={1} max={15} value={epochsToRun} disabled={!!progress} onChange={e => setEpochsToRun(Number(e.target.value))} />
        </label>
        <button type="button" className="btn btn-sm" disabled={!!progress} onClick={reset}>↺ Reset</button>
        <span className="muted vit-note">{modelRef.current ? `${modelRef.current.variables.reduce((n, v) => n + v.size, 0).toLocaleString('en')} weights` : ''} · 2 blocks · 2 heads · 32-number tokens</span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="dl-grid vit-charts">
        <div className="demo-panel">
          <h3>Validation accuracy</h3>
          <LineChart title="Validation accuracy per epoch" series={[{ label: 'ViT', color: '#34d399', values: accuracy }]}
            xStart={1} xLabel="epoch" yMax={1} formatY={v => `${Math.round(v * 100)}%`} height={170} />
        </div>
        <div className="demo-panel">
          <h3>Training loss</h3>
          <LineChart title="Training loss, averaged over 20 batches" series={[{ label: 'loss', color: '#f87171', values: smooth(losses, 20) }]}
            xLabel="× 20 batches" yMin={0} height={170} />
        </div>
      </div>

      {!result && <p className="muted">Press Train, then pick a digit to see where the transformer looks.</p>}
      {result && overlay && (
        <>
          <div className="demo-controls">
            <DigitPicker split={data.test} value={index} onChange={i => { setIndex(i); setQuery(null); }} />
          </div>
          <div className="demo-controls">
            <span className="demo-label">Attention:</span>
            {views.map(({ label, v }) => (
              <button key={label} type="button" aria-pressed={same(view, v)} className={`btn btn-sm${same(view, v) ? ' btn-primary' : ''}`}
                onClick={() => setView(v)}>{label}</button>
            ))}
          </div>
          <div className="vit-layout">
            <div className="demo-panel">
              <h3>{query === null ? 'What [CLS] attends to' : `What patch ${query + 1} attends to`}</h3>
              <PixelCanvas image={overlay} size={DISPLAY} label="Attention weights over the 16 patches of the digit">
                <div className="vit-patches">
                  {Array.from({ length: PATCHES }, (_, p) => (
                    <button key={p} type="button" className={`vit-patch${query === p ? ' vit-patch-active' : ''}`}
                      aria-label={`Patch ${p + 1}: ${weights ? (weights[p] * 100).toFixed(1) : 0}% attention`}
                      onClick={() => setQuery(query === p ? null : p)}>
                      <span>{weights ? Math.round(weights[p] * 100) : ''}</span>
                    </button>
                  ))}
                </div>
              </PixelCanvas>
              <p className="caption">{query === null ? 'Click a patch to use it as the query.' : 'Click it again to go back to [CLS].'}
                {view.kind === 'rollout' && query !== null ? ' (Rollout is for [CLS]; showing the last block.)' : ''}</p>
            </div>
            <div className="demo-panel vit-pred">
              <ProbBars probs={result.probs} trueLabel={data.test.labels[index]} title="ViT prediction" />
            </div>
            <div className="demo-panel">
              <h3>Learned position embeddings</h3>
              <div className="vit-sim" role="img" aria-label="For each patch, a 4 by 4 map of how similar its position embedding is to every other patch's">
                {Array.from({ length: PATCHES }, (_, p) => (
                  <PixelCanvas key={p} image={rgbImage(GRID, GRID, i => divergingColor(result.similarity[p * PATCHES + i]))} size={48} label="" />
                ))}
              </div>
              <p className="caption">Tile (r, c): cosine similarity of patch (r, c)'s position embedding to every patch. Red = similar.</p>
            </div>
          </div>
        </>
      )}
    </>
  );
}
