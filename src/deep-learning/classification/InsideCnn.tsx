import { Suspense, use, useDeferredValue, useMemo, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import insightCode from './insight.ts?raw';
import pythonCode from './lenet.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { rgbImage } from '../../lib/image';
import { DIGITS } from './constants';
import { interleaveByLabel } from './evaluation';
import {
  activations, conv1Filters, divergingColor, occlusionColor, occlusionMap, patchPositions,
  type FeatureLayer, type FeatureMaps, type OcclusionView,
} from './insight';
import { predictDigit } from './lenet';
import type { MnistSplit } from '../shared/mnist';

/**
 * The weights to look at — a frozen copy of the model, refreshed after every epoch —
 * plus a version number that changes whenever they do.
 */
export interface TrainedModel {
  model: tf.LayersModel;
  version: number;
}

interface InsideCnnProps {
  trained: TrainedModel;
  /** The last drawing classified in the Prediction tab (28×28, values 0–1). */
  drawing: Float32Array | null;
  /** Held-out test split to step through. */
  test: MnistSplit | null;
  /** Training is running: the views refresh after every epoch. */
  training: boolean;
}

type Source = 'drawing' | 'test';

const SIZE = 28;
const STRIDE = 2;
/** Even patch sizes only: 7 settings instead of 13, each one big forward pass. */
const PATCH = { min: 2, max: 14, step: 2 };
const INPUT_PX = 168;
const FILTER_PX = 60;
const OCCLUSION_PX = 224;
const RAMP_PX = 120;
const RAMP_STEPS = 65; // odd, so t = 0 (black) is one of the samples

/** Tile size on screen and a one-line description of each feature layer. */
const LAYERS: Record<FeatureLayer, { tile: number; what: string }> = {
  conv1: { tile: 72, what: 'input ⊛ each 5×5 filter, + bias, then ReLU' },
  pool1: { tile: 48, what: '2×2 average of conv1 — half the size' },
  conv2: { tile: 40, what: '16 filters of 5×5×6, each mixing all six pool1 maps' },
  pool2: { tile: 32, what: '2×2 average — these 256 numbers feed the dense layers' },
};

const finite = (v: number) => (Number.isFinite(v) ? v : 0);

/** 28×28 values in [0, 1] → grayscale image. */
function digitImage(pixels: Float32Array): ImageData {
  return rgbImage(SIZE, SIZE, i => {
    const v = pixels[i] * 255;
    return [v, v, v];
  });
}

/** One channel of a layer: black = 0, white = `max`, the largest value in the whole layer. */
function channelImage(maps: FeatureMaps, channel: number, max: number): ImageData {
  const offset = channel * maps.height * maps.width;
  return rgbImage(maps.width, maps.height, i => {
    const v = max > 0 ? (finite(maps.data[offset + i]) / max) * 255 : 0;
    return [v, v, v];
  });
}

/** The occlusion map in colour, with the digit as a light veil on top (see occlusionColor). */
function occlusionImage(input: Float32Array, map: Float32Array, view: OcclusionView, scale: number): ImageData {
  return rgbImage(SIZE, SIZE, i => occlusionColor(scale > 0 ? finite(map[i]) / scale : 0, input[i], view));
}

/** A colour bar for t = from … 1, drawn with the same colour function as the image it explains. */
function rampImage(colour: (t: number) => [number, number, number], from: number): ImageData {
  return rgbImage(RAMP_STEPS, 6, i => colour(from + ((1 - from) * (i % RAMP_STEPS)) / (RAMP_STEPS - 1)));
}

const percent = (p: number) => (Number.isFinite(p) ? `${(p * 100).toFixed(1)}%` : '—');

/** Resolves once the browser has painted the next frame. */
const afterNextPaint = () => new Promise<void>(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));

/** Everything one occlusion map depends on, plus what its labels say about it. */
interface OcclusionRequest {
  trained: TrainedModel;
  input: Float32Array;
  classIndex: number;
  patch: number;
  /** P(classIndex) for the untouched input, and how the input is called on screen. */
  probability: number | null;
  inputName: string;
  /** Resolves after the frame showing the new control state has been painted. */
  painted: Promise<void>;
}

function computeOcclusion({ trained, input, classIndex, patch }: OcclusionRequest) {
  const map = occlusionMap(trained.model, input, classIndex, { patch, stride: STRIDE });
  let maxDrop = 0, maxAbs = 0;
  for (const v of map) {
    maxDrop = Math.max(maxDrop, finite(v));
    maxAbs = Math.max(maxAbs, Math.abs(finite(v)));
  }
  return { input, map, maxDrop, maxAbs };
}

interface OcclusionPanelProps {
  /** The map to show (deferred: it may lag behind the controls), or null for none yet. */
  request: OcclusionRequest | null;
  /** The controls ask for a newer map than `request`. */
  pending: boolean;
  hasInput: boolean;
  view: OcclusionView;
}

function OcclusionPanel({ request, pending, hasInput, view }: OcclusionPanelProps) {
  // Suspend until the browser has painted the new slider / button state (React keeps the
  // previous map on screen meanwhile); only then run the big forward pass.
  if (request) use(request.painted);
  const occlusion = useMemo(() => (request ? computeOcclusion(request) : null), [request]);
  const image = useMemo(() => {
    if (!occlusion) return null;
    return occlusionImage(occlusion.input, occlusion.map, view, view === 'heat' ? occlusion.maxDrop : occlusion.maxAbs);
  }, [occlusion, view]);
  const ramps = useMemo(() => ({
    heat: rampImage(t => occlusionColor(t, 0, 'heat'), 0),
    signed: rampImage(t => occlusionColor(t, 0, 'signed'), -1),
  }), []);
  // Labels describe the map on screen, which may lag behind the controls (see `pending`).
  const d = request ? DIGITS[request.classIndex] : null;
  const probability = request?.probability ?? null;

  return (
    <div className={`demo-panel cnn-occlusion-panel${pending ? ' cnn-pending' : ''}`} aria-busy={pending}>
      <h3 id="cnn-occ-heading">Occlusion sensitivity{d !== null && ` · class ${d}`}</h3>
      <PixelCanvas image={image} size={OCCLUSION_PX} placeholder={hasInput ? '⏳ Computing…' : 'No input'}
        label={request ? `Occlusion map of ${request.inputName} for class ${d}: coloured where hiding pixels changes the probability` : 'Occlusion map'}>
        {pending && <span className="cnn-pending-badge">⏳ updating…</span>}
      </PixelCanvas>
      {occlusion && (
        <>
          <div className="cnn-legend">
            <span className="cnn-legend-title">Drop in P({d}):</span>
            {view === 'heat' ? (
              <>
                <span className="mono">0</span>
                <PixelCanvas image={ramps.heat} size={RAMP_PX} pixelated={false} className="cnn-ramp"
                  label="Colour scale: black = no drop, through red and yellow to the largest drop" />
                <span className="mono">{(occlusion.maxDrop * 100).toFixed(1)} pts</span>
              </>
            ) : (
              <>
                <span className="mono">−{(occlusion.maxAbs * 100).toFixed(1)}</span>
                <PixelCanvas image={ramps.signed} size={RAMP_PX} pixelated={false} className="cnn-ramp"
                  label="Colour scale: blue = negative drop (P rises), black = 0, red = positive drop (P falls)" />
                <span className="mono">+{(occlusion.maxAbs * 100).toFixed(1)} pts</span>
              </>
            )}
          </div>
          <p className="caption">
            {view === 'heat'
              ? `Brighter = bigger drop in P(${d}) when hidden`
              : `Red (+) = hiding lowers P(${d}) · blue (−) = hiding raises it`}
            {' '}· pts = percentage points · the digit shows as a light veil
            {probability !== null && ` · P(${d}) = ${percent(probability)}`}
          </p>
        </>
      )}
    </div>
  );
}

export function InsideCnn({ trained, drawing, test, training }: InsideCnnProps) {
  const [source, setSource] = useState<Source>(() => (drawing ? 'drawing' : 'test'));
  const [testPos, setTestPos] = useState(0);
  const [patch, setPatch] = useState(6);
  const [view, setView] = useState<OcclusionView>('heat');
  /** Class whose occlusion map is shown; null = the predicted class. */
  const [chosenClass, setChosenClass] = useState<number | null>(null);
  const [showCode, setShowCode] = useState(false);

  // Step through the test split as 0, 1, …, 9, 0, 1, … rather than 100 zeros first.
  const order = useMemo(() => (test ? interleaveByLabel(test.labels) : []), [test]);
  const fromDrawing = source === 'drawing' && drawing !== null;
  const testIndex = !fromDrawing && order.length > 0 ? order[testPos] : -1;
  const trueLabel = test && testIndex >= 0 ? test.labels[testIndex] : null;

  const input = useMemo(() => {
    if (fromDrawing) return drawing;
    if (!test || testIndex < 0) return null;
    return test.images.slice(testIndex * SIZE * SIZE, (testIndex + 1) * SIZE * SIZE);
  }, [fromDrawing, drawing, test, testIndex]);
  const inputImage = useMemo(() => (input ? digitImage(input) : null), [input]);

  // ---- Everything below is recomputed only when the weights (trained), the input or a setting changes.
  const probs = useMemo(() => (input ? predictDigit(trained.model, input) : null), [trained, input]);
  let predicted = 0;
  if (probs) for (let d = 1; d < probs.length; d++) if (probs[d] > probs[predicted]) predicted = d;
  const explained = chosenClass ?? predicted;
  const probability = probs ? probs[explained] : null;
  const inputName = fromDrawing ? 'your drawing' : `test digit #${testPos + 1}`;

  const filters = useMemo(() => {
    const kernels = conv1Filters(trained.model);
    // One shared scale, so the six filters can be compared with each other.
    let scale = 0;
    for (const k of kernels) for (const w of k) scale = Math.max(scale, Math.abs(finite(w)));
    const images = kernels.map(k => rgbImage(5, 5, i => divergingColor(scale > 0 ? k[i] / scale : 0)));
    return { images, scale };
  }, [trained]);

  const layers = useMemo(() => {
    if (!input) return null;
    return activations(trained.model, input).map(maps => {
      let max = 0;
      for (const v of maps.data) max = Math.max(max, finite(v));
      const images = Array.from({ length: maps.channels }, (_, c) => channelImage(maps, c, max));
      return { maps, max, images };
    });
  }, [trained, input]);

  // The occlusion map takes one big forward pass (65–197 images), so it is computed from a
  // *deferred* copy of its inputs: the slider, buttons and stepper update at once, the old map
  // stays on screen (dimmed), and the new one is computed once that frame has been painted.
  // Quick successive changes are merged: only the latest request is computed.
  const occlusionRequest = useMemo<OcclusionRequest | null>(
    () => (input ? { trained, input, classIndex: explained, patch, probability, inputName, painted: afterNextPaint() } : null),
    [trained, input, explained, patch, probability, inputName],
  );
  const shownRequest = useDeferredValue(occlusionRequest, null);
  const positions = patchPositions(SIZE, patch, STRIDE).length;

  const filterRamp = useMemo(() => rampImage(divergingColor, -1), []);

  const pickSource = (s: Source) => {
    setSource(s);
    setChosenClass(null);
  };
  const step = (delta: number) => {
    setTestPos(p => (p + delta + order.length) % order.length);
    setChosenClass(null);
  };

  return (
    <div className="cnn-inside">
      <div className="demo-controls">
        <span className="demo-label">Input:</span>
        <div className="segmented cnn-segmented" role="group" aria-label="Input to inspect">
          {/* aria-disabled rather than disabled: it stays focusable, so the hint can be reached */}
          <button type="button" aria-pressed={fromDrawing} aria-disabled={!drawing}
            aria-describedby={drawing ? undefined : 'cnn-drawing-hint'}
            className={`btn btn-sm${fromDrawing ? ' btn-primary' : ''}`} onClick={() => { if (drawing) pickSource('drawing'); }}>
            ✏️ My drawing
          </button>
          <button type="button" aria-pressed={!fromDrawing} disabled={!test}
            className={`btn btn-sm${!fromDrawing ? ' btn-primary' : ''}`} onClick={() => pickSource('test')}>
            🔢 Test digit
          </button>
        </div>
        {!fromDrawing && order.length > 0 && (
          <div className="cnn-stepper">
            <button type="button" className="btn btn-sm" aria-label="Previous test digit" onClick={() => step(-1)}>◀</button>
            <span className="mono cnn-step-label" aria-live="polite">#{testPos + 1} / {order.length}</span>
            <button type="button" className="btn btn-sm" aria-label="Next test digit" onClick={() => step(1)}>▶</button>
          </div>
        )}
        {!drawing && (
          <span id="cnn-drawing-hint" className="caption cnn-hint">✏️ My drawing: draw a digit and click Classify in the Prediction tab first.</span>
        )}
        <span className="ml-auto">
          <ShowCodeButton code={insightCode} title="classification/insight.ts" python={{ code: pythonCode, title: 'classification/lenet.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {/* Both messages share one grid cell and only one is visible, so the page never jumps. */}
      <p className={`cnn-status${training ? ' cnn-status-training' : ''}`} role="status">
        <span className={training ? undefined : 'cnn-status-off'}>
          ⏳ Training… everything here uses the weights of the last finished epoch and updates after each one.
        </span>
        <span className={training ? 'cnn-status-off' : undefined}>
          Everything here uses the current trained weights; train more and it updates after every epoch.
        </span>
      </p>

      <div className="cnn-row">
        <div className="demo-panel cnn-input-panel">
          <h3>{fromDrawing ? 'Your drawing' : 'Test digit'} · 28×28</h3>
          <PixelCanvas image={inputImage} size={INPUT_PX} label={`The 28×28 input: ${inputName}`} placeholder="No input" />
          {probs && (
            <div className="cnn-prediction" aria-live="polite">
              <span className="muted">Predicted</span>
              <span className="cnn-pred-digit">{DIGITS[predicted]}</span>
              <span className="cnn-pred-conf">{percent(probs[predicted])}</span>
            </div>
          )}
          {trueLabel !== null && (
            <p className="cnn-true-label">
              True label <strong className="mono">{DIGITS[trueLabel]}</strong>{' '}
              {trueLabel === predicted ? <span className="text-success">✓ correct</span> : <span className="text-error">✗ wrong</span>}
            </p>
          )}
        </div>

        <div className="demo-panel cnn-filters-panel">
          <h3>First-layer filters · conv1 · 6 × 5×5</h3>
          <div className="cnn-filters">
            {filters.images.map((img, i) => (
              <figure key={i} className="cnn-tile">
                <PixelCanvas image={img} size={FILTER_PX} label={`conv1 filter ${i + 1}: 5×5 learned weights`} />
                <figcaption className="caption mono">{i + 1}</figcaption>
              </figure>
            ))}
          </div>
          <div className="cnn-legend">
            <span className="mono">−{filters.scale.toFixed(2)}</span>
            <PixelCanvas image={filterRamp} size={RAMP_PX} pixelated={false} className="cnn-ramp"
              label="Colour scale: blue = negative weight, black = 0, red = positive" />
            <span className="mono">+{filters.scale.toFixed(2)}</span>
          </div>
          <p className="caption">Blue = negative weight · black = 0 · red = positive</p>
        </div>

        <div className="demo-explanation cnn-explain">
          <h4>Filters: learned convolution kernels</h4>
          <p>
            Each 5×5 filter slides over the image exactly like the hand-made kernels in the <strong>Convolution</strong> section
            (Sobel, blur, sharpen…). The difference: nobody designed these 25 weights — gradient descent tuned them
            so the network classifies the training digits well.
          </p>
          <p>
            A red band next to a blue band responds to <strong>edges</strong> of that orientation (like Sobel); a red blob responds
            to <strong>strokes</strong>. Don't expect textbook detectors here, though: these filters start as random noise, and
            training mostly raises all 25 weights of a filter together (it drifts towards red and responds more to ink in general)
            while its pattern stays the random one it started with. The later layers learn to use whatever these filters pick up,
            which is enough for MNIST; clean edge and colour detectors appear in large networks trained on photos.
          </p>
        </div>
      </div>

      <section className="panel cnn-activations" aria-labelledby="cnn-act-heading">
        <h3 className="panel-heading" id="cnn-act-heading">Activation maps for {inputName}</h3>
        {layers ? layers.map(({ maps, max, images }) => (
          <div key={maps.name} className="cnn-layer">
            <div className="cnn-layer-head">
              <span className="mono cnn-layer-name">{maps.name}</span>
              <span className="mono cnn-layer-shape">{maps.height}×{maps.width}×{maps.channels}</span>
              <span className="caption cnn-layer-what">{LAYERS[maps.name].what} · white = {max.toFixed(2)}</span>
            </div>
            <div className={`cnn-tiles cnn-tiles-${maps.channels}`}>
              {images.map((img, c) => (
                <figure key={c} className="cnn-tile">
                  <PixelCanvas image={img} size={LAYERS[maps.name].tile}
                    label={`${maps.name} channel ${c + 1} of ${maps.channels}, ${maps.height}×${maps.width}`} />
                  <figcaption className="caption mono">{c + 1}</figcaption>
                </figure>
              ))}
            </div>
          </div>
        )) : <p className="muted">No input to show.</p>}
        <div className="cnn-legend cnn-legend-start">
          <span className="mono">0</span>
          <span className="cnn-legend-bar cnn-legend-gray" aria-hidden="true" />
          <span className="mono">layer max</span>
          <span className="caption">Each layer is scaled to its own maximum (shown as “white = …”), so its channels can be compared.</span>
        </div>
        <div className="cnn-text">
          <p>
            Each tile is one <strong>channel</strong>: where in the image one filter responded (bright) or not (black — ReLU turns negative
            responses into 0). Channel <em>k</em> of conv1 is the input filtered by filter <em>k</em> above.
          </p>
          <p>
            Going deeper the maps get <strong>smaller</strong> (pooling halves them: 24 → 12, 8 → 4) but <strong>more numerous</strong> (6 → 16),
            and more <strong>abstract</strong>: a conv2 channel combines all six pool1 maps, so it reacts to arrangements of strokes —
            corners, curves, loops — rather than to plain edges. They become hard for us to read, yet they are what the dense layers
            use to tell a 3 from an 8.
          </p>
        </div>
      </section>

      <section className="cnn-occlusion" aria-labelledby="cnn-occ-heading">
        <Suspense fallback={<div className="demo-panel cnn-occlusion-panel" aria-busy="true">⏳ Computing…</div>}>
          <OcclusionPanel request={shownRequest} pending={shownRequest !== occlusionRequest} hasInput={input !== null} view={view} />
        </Suspense>

        <div className="cnn-occlusion-side">
          <div className="demo-slider-section cnn-occlusion-controls">
            <div className="demo-slider-row">
              <label htmlFor="cnn-patch" className="demo-slider-label">Patch: <strong>{patch}×{patch}</strong> px</label>
              <input id="cnn-patch" type="range" className="demo-slider" min={PATCH.min} max={PATCH.max} step={PATCH.step}
                value={patch} onChange={e => setPatch(Number(e.target.value))} />
            </div>
            <div className="cnn-control-row cnn-class-row">
              <span className="demo-label" id="cnn-class-label">Explain class:</span>
              <div className="segmented cnn-segmented" role="group" aria-labelledby="cnn-class-label">
                {DIGITS.map((d, i) => (
                  <button key={d} type="button" aria-pressed={explained === i}
                    aria-label={`Class ${d}${i === predicted ? ' (predicted)' : ''}`}
                    className={`btn btn-sm cnn-class-btn${explained === i ? ' btn-primary' : ''}${i === predicted ? ' cnn-class-predicted' : ''}`}
                    onClick={() => setChosenClass(i)}>
                    {d}
                  </button>
                ))}
              </div>
            </div>
            <div className="cnn-control-row">
              <span className="demo-label">Colours:</span>
              <div className="segmented cnn-segmented" role="group" aria-label="Occlusion map colours">
                <button type="button" aria-pressed={view === 'heat'} className={`btn btn-sm${view === 'heat' ? ' btn-primary' : ''}`}
                  onClick={() => setView('heat')}>🔥 Drops only</button>
                <button type="button" aria-pressed={view === 'signed'} className={`btn btn-sm${view === 'signed' ? ' btn-primary' : ''}`}
                  onClick={() => setView('signed')}>± Signed</button>
              </div>
            </div>
            <p className="caption">
              {positions}×{positions} = {positions * positions} occluded copies (stride {STRIDE} px) + the original, classified in one batch.
              The underlined digit is the prediction.
            </p>
          </div>

          <div className="demo-explanation">
            <h4>Which pixels does the prediction rely on?</h4>
            <p>
              Slide a black {patch}×{patch} square over the digit, setting those pixels to 0 (the background), and ask the network again.
              Every pixel gets the <strong>average drop in P({DIGITS[explained]})</strong> over all the squares that covered it.
              Positive (bright / red) = hiding these pixels hurts: the network relies on them. Negative (blue) = hiding them
              <em> raises</em> the probability: they look like evidence for another digit.
            </p>
            <h4>Limits</h4>
            <ul className="cnn-limits">
              <li>It can only remove ink. Hiding background changes nothing, so the map cannot show that “no ink here” is evidence
                (the hole of a 0, the open top of a 4). Background next to a stroke is coloured only because the same squares also hid the stroke.</li>
              <li>An occluded digit is an image the network never saw: breaking a loop can turn a 9 into a 4, so a drop mixes
                “evidence removed” with “new evidence created”.</li>
              <li>One patch at a time: if two strokes are each enough on their own, hiding either does little, so redundant evidence looks unimportant.</li>
              <li>The patch size trades sharpness for strength — small patches barely matter, large ones blur the map.
                And it explains this one input, not the network in general.</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
