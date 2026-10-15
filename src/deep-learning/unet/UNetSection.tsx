import { useEffect, useMemo, useRef, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import unetCode from './unet.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import { applyThreshold, histogram256, otsuThreshold } from '../../image-processing/segmentation/otsu';
import { floatImage, heatImage } from '../shared/render';
import { DEFAULT_DIFFICULTY, makeDataset, makeSample, maskIoU, type Difficulty, type SegSample } from './shapes';
import { buildUNet, predictMasks, trainEpoch, UNET_SIZE } from './unet';
import '../shared/dl-shared.css';
import './unet.css';

const N = UNET_SIZE;
const TRAIN_COUNT = 800;
const VAL_COUNT = 100;
const GALLERY = 5;
const CELL = 112;

/** Otsu's global threshold, as in the Segmentation lesson. */
function otsu(image: Float32Array): Uint8Array {
  const pixels = Float64Array.from(image, v => v * 255);
  return applyThreshold(pixels, otsuThreshold(histogram256(pixels)));
}

interface EpochLog { loss: number; iou: number }

export function UNetSection() {
  const [skips, setSkips] = useState(true);
  const [epochsToRun, setEpochsToRun] = useState(10);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [status, setStatus] = useState<'idle' | 'training'>('idle');
  const [batch, setBatch] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [testSeed, setTestSeed] = useState(1);
  const [difficulty, setDifficulty] = useState<Difficulty>(DEFAULT_DIFFICULTY);
  const [showCode, setShowCode] = useState(false);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const train = useMemo(() => makeDataset(TRAIN_COUNT, N, seededRandom(1)), []);
  const val = useMemo(() => makeDataset(VAL_COUNT, N, seededRandom(2)), []);
  const gallery = useMemo<SegSample[]>(() => {
    const random = seededRandom(1000 + testSeed);
    return Array.from({ length: GALLERY }, () => makeSample(N, random, difficulty));
  }, [testSeed, difficulty]);

  const otsuVal = useMemo(() => {
    let sum = 0;
    for (let n = 0; n < VAL_COUNT; n++) {
      const img = val.images.subarray(n * N * N, (n + 1) * N * N);
      sum += maskIoU(otsu(img), val.masks.subarray(n * N * N, (n + 1) * N * N));
    }
    return sum / VAL_COUNT;
  }, [val]);

  useEffect(() => () => {
    abortRef.current?.abort();
    modelRef.current?.dispose();
  }, []);

  const reset = (nextSkips = skips) => {
    abortRef.current?.abort();
    modelRef.current?.dispose();
    modelRef.current = null;
    setSkips(nextSkips);
    setHistory([]);
    setVersion(v => v + 1);
  };

  const start = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    modelRef.current ??= buildUNet({ skips });
    const model = modelRef.current;
    setStatus('training');
    try {
      for (let e = 0; e < epochsToRun && !controller.signal.aborted; e++) {
        const loss = await trainEpoch(model, train.images, train.masks, train.count, (b, total) => setBatch(`batch ${b}/${total}`));
        if (controller.signal.aborted || modelRef.current !== model) break;
        const p = predictMasks(model, val.images, VAL_COUNT);
        let iou = 0;
        for (let n = 0; n < VAL_COUNT; n++) iou += maskIoU(p.subarray(n * N * N, (n + 1) * N * N), val.masks.subarray(n * N * N, (n + 1) * N * N));
        setHistory(h => [...h, { loss, iou: iou / VAL_COUNT }]);
        setVersion(v => v + 1);
      }
    } catch (err) {
      // Reset or leaving the page disposes the model under a running epoch: that is a normal stop.
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abortRef.current === controller) setStatus('idle');
      setBatch(null);
    }
  };

  const predictions = useMemo(() => {
    const model = modelRef.current;
    if (!model || version < 0) return null;
    const packed = new Float32Array(GALLERY * N * N);
    gallery.forEach((s, k) => packed.set(s.image, k * N * N));
    return predictMasks(model, packed, GALLERY);
  }, [gallery, version]);

  const setD = (key: keyof Difficulty, v: number) => setDifficulty(d => ({ ...d, [key]: v }));

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" disabled={status === 'training'} onClick={start}>
          {status === 'training' ? `Training… epoch ${history.length + 1}${batch ? `, ${batch}` : ''}` : history.length ? `▶ Train ${epochsToRun} more epochs` : `▶ Train ${epochsToRun} epochs`}
        </button>
        {status === 'training' && <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>⏹ Stop</button>}
        <label className="dl-slider">Epochs <strong>{epochsToRun}</strong>
          <input type="range" min={1} max={30} value={epochsToRun} onChange={e => setEpochsToRun(Number(e.target.value))} />
        </label>
        <label className="dl-slider">
          <input type="checkbox" checked={skips} disabled={status === 'training'} onChange={e => reset(e.target.checked)} /> Skip connections
        </label>
        <button type="button" className="btn btn-sm" onClick={() => reset()}>↺ Reset network</button>
        <span className="ml-auto">
          <ShowCodeButton code={unetCode} title="unet/unet.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="dl-grid unet-top">
        <div className="demo-panel">
          <h3>Validation IoU ({VAL_COUNT} images)</h3>
          <LineChart
            title="Mean intersection over union on validation images per epoch, against Otsu's threshold"
            series={[
              { label: skips ? 'U-Net' : 'without skips', color: '#34d399', values: history.map(h => h.iou) },
              { label: `Otsu threshold (${otsuVal.toFixed(2)})`, color: '#fb923c', values: Array(Math.max(2, history.length)).fill(otsuVal), dashed: true },
            ]}
            yMin={0}
            yMax={1}
            xStart={1}
            xLabel="epoch"
          />
        </div>
        <div className="demo-panel">
          <h3>Training loss (binary cross-entropy)</h3>
          <LineChart title="Training loss per epoch" series={[{ label: 'loss', color: '#f87171', values: history.map(h => h.loss) }]} xStart={1} xLabel="epoch" yMin={0} />
        </div>
      </div>

      <div className="demo-controls">
        <span className="demo-label">Test images:</span>
        <button type="button" className="btn btn-sm" onClick={() => setTestSeed(s => s + 1)}>🎲 New images</button>
        <label className="dl-slider">Uneven lighting <strong>{difficulty.lighting.toFixed(2)}</strong>
          <input type="range" min={0} max={0.8} step={0.05} value={difficulty.lighting} onChange={e => setD('lighting', Number(e.target.value))} />
        </label>
        <label className="dl-slider">Noise <strong>{difficulty.noise.toFixed(2)}</strong>
          <input type="range" min={0} max={0.2} step={0.01} value={difficulty.noise} onChange={e => setD('noise', Number(e.target.value))} />
        </label>
        <label className="dl-slider">Clutter lines <strong>{difficulty.clutter}</strong>
          <input type="range" min={0} max={8} value={difficulty.clutter} onChange={e => setD('clutter', Number(e.target.value))} />
        </label>
      </div>

      <div className="unet-gallery" role="table" aria-label="Test images with their segmentations">
        <div className="unet-row unet-head" role="row">
          {['Input', 'Ground truth', 'Otsu threshold', 'U-Net probability', 'U-Net mask'].map(h => <span key={h} role="columnheader">{h}</span>)}
        </div>
        {gallery.map((s, k) => {
          const o = otsu(s.image);
          const p = predictions?.subarray(k * N * N, (k + 1) * N * N);
          return (
            <div key={k} className="unet-row" role="row">
              <Cell image={floatImage(s.image, N, N)} label="Input" />
              <Cell image={floatImage(s.mask, N, N)} label="Ground truth mask" />
              <Cell image={floatImage(o, N, N)} label="Otsu mask" caption={`IoU ${maskIoU(o, s.mask).toFixed(2)}`} />
              <Cell image={p ? heatImage(p, N, N, 1) : null} label="U-Net probability" caption={p ? '' : 'train first'} />
              <Cell image={p ? floatImage(Array.from(p, v => (v >= 0.5 ? 1 : 0)), N, N) : null} label="U-Net mask"
                caption={p ? `IoU ${maskIoU(p, s.mask).toFixed(2)}` : ''} />
            </div>
          );
        })}
      </div>

      <div className="demo-explanation">
        <h4>Segmentation as per-pixel classification</h4>
        <p>Otsu's method (Segmentation lesson) picks one brightness threshold for the whole image. Here that cannot work: the
        lighting makes one side of the background brighter than the objects on the other side, and the clutter lines are as
        bright as objects. Deciding whether a pixel belongs to an object needs its <em>neighbourhood</em>: is it part of a
        filled blob or a thin line?</p>
        <p>A <strong>U-Net</strong> classifies every pixel at once. Its encoder pools the image down to 8×8 so each unit sees a
        large area (see the Receptive field lesson); its decoder upsamples back to 32×32. The <strong>skip connections</strong>
        copy the encoder's full-resolution features across, so the output keeps sharp edges. Untick them and retrain: the
        network still finds the objects but their outlines come out blurred and blocky. Then raise the lighting, noise or
        clutter beyond what it was trained on and watch where it starts to fail.</p>
      </div>
    </div>
  );
}

function Cell({ image, label, caption }: { image: ImageData | null; label: string; caption?: string }) {
  return (
    <div className="unet-cell" role="cell">
      <PixelCanvas image={image} size={CELL} label={label} placeholder="—" />
      {caption !== undefined && <span className="caption">{caption || ' '}</span>}
    </div>
  );
}
