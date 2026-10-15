import { useEffect, useMemo, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import saliencyCode from './saliency.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import { occlusionMap } from '../classification/insight';
import { buildLeNet, predictDigit } from '../classification/lenet';
import { DigitPicker } from '../shared/DigitPicker';
import { LeNetGate } from '../shared/LeNetGate';
import type { SharedLeNet } from '../shared/modelStore';
import type { MnistSplit } from '../shared/mnist';
import { ProbBars } from '../shared/ProbBars';
import { floatImage, heatImage, heatOverInput } from '../shared/render';
import { useMnist } from '../shared/useMnist';
import { gradCam, smoothGrad, vanillaGradient } from './saliency';
import '../shared/dl-shared.css';
import './saliency.css';

const DISPLAY = 168;

export function SaliencySection() {
  const [showCode, setShowCode] = useState(false);
  const { data, error } = useMnist();
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Four ways to ask a trained network "which pixels made you say that?"</span>
        <span className="ml-auto">
          <ShowCodeButton code={saliencyCode} title="saliency/saliency.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      <LeNetGate purpose="explains a trained classifier's predictions">
        {lenet => (data ? <Explain lenet={lenet} test={data.test} /> : <p className="dl-status">Loading MNIST…</p>)}
      </LeNetGate>
      <div className="demo-explanation">
        <h4>Gradients, Grad-CAM and occlusion</h4>
        <p><strong>Vanilla gradient</strong>: how much log p(class) changes when one pixel changes. Cheap (one backward pass)
        but noisy, because the gradient of a ReLU network jumps around between nearby inputs. <strong>SmoothGrad</strong>{' '}
        averages the gradient over noisy copies of the input to wash that out.</p>
        <p><strong>Grad-CAM</strong> works one level up: it weights each of the last convolution's 16 feature maps by the mean
        gradient flowing into it and adds them up. The result is coarse (8×8, upsampled) but class-specific and much more
        readable. <strong>Occlusion</strong> needs no gradients at all: it hides patches and measures the drop in probability.</p>
        <p>Switch the class to explain: a good map should change, highlighting the strokes that would argue for a different
        digit. Then try the <strong>sanity check</strong>: an explanation method that gives a similar picture for an untrained
        network is showing you edges in the input, not what the model learned (Adebayo et al., 2018).</p>
      </div>
    </div>
  );
}

function Explain({ lenet, test }: { lenet: SharedLeNet; test: MnistSplit }) {
  const [index, setIndex] = useState(200);
  const [explainClass, setExplainClass] = useState<number | 'predicted'>('predicted');
  const [untrained, setUntrained] = useState(false);
  const [randomModel, setRandomModel] = useState<tf.LayersModel | null>(null);

  // The sanity check's untrained network, built on first use and freed with the view.
  useEffect(() => {
    if (!untrained || randomModel) return;
    setRandomModel(buildLeNet());
  }, [untrained, randomModel]);
  useEffect(() => () => { randomModel?.dispose(); }, [randomModel]);

  const image = test.images.subarray(index * 784, (index + 1) * 784);
  const label = test.labels[index];

  const result = useMemo(() => {
    const model = untrained ? randomModel : lenet.model;
    if (!model) return null;
    const probs = predictDigit(model, image);
    const predicted = probs.indexOf(Math.max(...probs));
    const cls = explainClass === 'predicted' ? predicted : explainClass;
    const occlusion = occlusionMap(model, image, cls).map(v => Math.max(0, v));
    const cam = gradCam(model, image, cls);
    return {
      probs, predicted, cls,
      maps: [
        { name: 'Vanilla gradient', values: vanillaGradient(model, image, cls) },
        { name: 'SmoothGrad', values: smoothGrad(model, image, cls, seededRandom(index)) },
        { name: 'Grad-CAM (conv2)', values: cam.map, coarse: cam.coarse },
        { name: 'Occlusion', values: occlusion },
      ],
    };
  }, [lenet, randomModel, untrained, image, index, explainClass]);

  return (
    <>
      <div className="demo-controls">
        <DigitPicker split={test} value={index} onChange={setIndex} />
      </div>
      <div className="demo-controls">
        <span className="demo-label">Explain the class:</span>
        <button type="button" aria-pressed={explainClass === 'predicted'} className={`btn btn-sm${explainClass === 'predicted' ? ' btn-primary' : ''}`}
          onClick={() => setExplainClass('predicted')}>predicted{result ? ` (${result.predicted})` : ''}</button>
        {Array.from({ length: 10 }, (_, d) => (
          <button key={d} type="button" aria-pressed={explainClass === d} className={`btn btn-sm${explainClass === d ? ' btn-primary' : ''}`}
            onClick={() => setExplainClass(d)}>{d}</button>
        ))}
        <label className="dl-slider sal-sanity">
          <input type="checkbox" checked={untrained} onChange={e => setUntrained(e.target.checked)} /> Sanity check: untrained network
        </label>
      </div>

      {result && (
        <div className="sal-row">
          <div className="demo-panel sal-input">
            <h3>Input (a {label})</h3>
            <PixelCanvas image={floatImage(image, 28, 28)} size={DISPLAY} label={`Test digit ${label}`} />
            <ProbBars probs={result.probs} trueLabel={label} title={untrained ? 'Untrained network' : 'Prediction'} />
          </div>
          {result.maps.map(m => (
            <div key={m.name} className="demo-panel">
              <h3>{m.name}</h3>
              <PixelCanvas image={heatOverInput(m.values, image, 28, 28)} size={DISPLAY} label={`${m.name} for class ${result.cls}`} />
              {m.coarse && (
                <div className="sal-coarse">
                  <PixelCanvas image={heatImage(m.coarse, 8, 8)} size={56} label="Grad-CAM before upsampling, 8 by 8" />
                  <span className="caption">8×8 before upsampling</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      {result && <p className="caption sal-note">Bright = the class {result.cls} score depends strongly on these pixels. Each map is scaled to its own maximum.</p>}
    </>
  );
}
