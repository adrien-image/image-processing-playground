import { useMemo, useState } from 'react';
import type * as tf from '@tensorflow/tfjs';
import attackCode from './attacks.ts?raw';
import pythonCode from './attacks.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { predictDigit } from '../classification/lenet';
import { DigitPicker } from '../shared/DigitPicker';
import { LeNetGate } from '../shared/LeNetGate';
import type { SharedLeNet } from '../shared/modelStore';
import type { MnistSplit } from '../shared/mnist';
import { ProbBars } from '../shared/ProbBars';
import { floatImage, signedImage } from '../shared/render';
import { useMnist } from '../shared/useMnist';
import { accuracyUnderAttack, attack } from './attacks';
import '../shared/dl-shared.css';
import './adversarial.css';

const DISPLAY = 196;
const SWEEP = Array.from({ length: 13 }, (_, i) => i * 0.025); // ε = 0 … 0.3
const SWEEP_IMAGES = 500;

type Mode = 'fgsm' | 'targeted' | 'pgd';

export function AdversarialSection() {
  const [showCode, setShowCode] = useState(false);
  const { data, error } = useMnist();
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Fool a trained digit classifier with changes too small to notice</span>
        <span className="ml-auto">
          <ShowCodeButton code={attackCode} title="adversarial/attacks.ts" python={{ code: pythonCode, title: 'adversarial/attacks.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      <LeNetGate purpose="attacks a trained classifier">
        {lenet => (data ? <Attack lenet={lenet} test={data.test} /> : <p className="dl-status">Loading MNIST…</p>)}
      </LeNetGate>
      <div className="demo-explanation">
        <h4>Turning the gradient against the network</h4>
        <p>Training computes how the loss changes with every <em>weight</em>. The same backpropagation also gives how it
        changes with every <em>pixel</em>. The <strong>fast gradient sign method</strong> nudges each pixel by ±ε in the
        direction that increases the loss of the true class:</p>
        <p className="formula"><code>x′ = clip(x + ε · sign(∇ₓ L(x, y)), 0, 1)</code></p>
        <p>No pixel moves by more than ε, but 784 small pushes all point the same way and add up. A <strong>targeted</strong>{' '}
        attack instead lowers the loss of a class you choose, and the <strong>iterative</strong> attack (PGD) takes several small
        steps, re-measuring the gradient each time, which needs a much smaller ε. The robustness curve shows the accuracy on
        500 test digits collapsing as ε grows. Adversarial training (adding attacked images to the training set) is the main
        defence, at the cost of some clean accuracy.</p>
      </div>
    </div>
  );
}

function Attack({ lenet, test }: { lenet: SharedLeNet; test: MnistSplit }) {
  const [index, setIndex] = useState(300);
  const [mode, setMode] = useState<Mode>('fgsm');
  const [epsilon, setEpsilon] = useState(0.1);
  const [target, setTarget] = useState(8);
  const [sweep, setSweep] = useState<{ version: number; values: number[] } | null>(null);

  const image = test.images.subarray(index * 784, (index + 1) * 784);
  const label = test.labels[index];
  const model: tf.LayersModel = lenet.model;
  const { version } = lenet;

  // `lenet` is a new object whenever the shared weights change, so results recompute then.
  const result = useMemo(() => {
    const model = lenet.model;
    const options = mode === 'targeted' ? { epsilon, target, steps: 10 } : mode === 'pgd' ? { epsilon, steps: 10 } : { epsilon };
    const adv = attack(model, image, label, options);
    const delta = adv.map((v, i) => v - image[i]);
    return { adv, delta, before: predictDigit(model, image), after: predictDigit(model, adv) };
  }, [lenet, image, label, mode, epsilon, target]);

  const fooled = argmax(result.after) !== label;
  const runSweep = () => setSweep({ version, values: SWEEP.map(e => accuracyUnderAttack(model, test.images, test.labels, e, SWEEP_IMAGES)) });

  return (
    <>
      <div className="demo-controls">
        <DigitPicker split={test} value={index} onChange={setIndex} />
      </div>
      <div className="demo-controls">
        <span className="demo-label">Attack:</span>
        {([['fgsm', 'FGSM (one step)'], ['targeted', 'Targeted'], ['pgd', 'Iterative (PGD, 10 steps)']] as const).map(([m, l]) => (
          <button key={m} type="button" aria-pressed={mode === m} className={`btn btn-sm${mode === m ? ' btn-primary' : ''}`} onClick={() => setMode(m)}>{l}</button>
        ))}
        {mode === 'targeted' && (
          <span className="adv-targets" role="group" aria-label="Target class">
            <span className="demo-label">make it a</span>
            {Array.from({ length: 10 }, (_, d) => (
              <button key={d} type="button" aria-pressed={target === d} disabled={d === label}
                className={`btn btn-sm${target === d ? ' btn-primary' : ''}`} onClick={() => setTarget(d)}>{d}</button>
            ))}
          </span>
        )}
        <label className="dl-slider">ε <strong>{epsilon.toFixed(3)}</strong>
          <input type="range" min={0} max={0.3} step={0.005} value={epsilon} onChange={e => setEpsilon(Number(e.target.value))} />
        </label>
      </div>

      <div className="dl-grid adv-grid">
        <div className="demo-panel">
          <h3>Original (a {label})</h3>
          <PixelCanvas image={floatImage(image, 28, 28)} size={DISPLAY} label={`Test digit ${label}`} />
          <ProbBars probs={result.before} trueLabel={label} title="Prediction" />
        </div>
        <div className="demo-panel adv-plus">
          <h3>+ perturbation (contrast boosted)</h3>
          <PixelCanvas image={signedImage(result.delta, 28, 28, Math.max(epsilon, 1e-6))} size={DISPLAY} label="Perturbation: red brighter, blue darker" />
          <p className="caption">red = brighter, blue = darker, every pixel by at most ε = {epsilon.toFixed(3)}</p>
        </div>
        <div className={`demo-panel${fooled ? ' adv-fooled' : ''}`}>
          <h3>= adversarial {fooled ? '(fooled!)' : '(still correct)'}</h3>
          <PixelCanvas image={floatImage(result.adv, 28, 28)} size={DISPLAY} label="Adversarial image" />
          <ProbBars probs={result.after} trueLabel={label} title="Prediction" />
        </div>
      </div>

      <div className="demo-panel">
        <h3>Robustness: accuracy under FGSM</h3>
        {sweep && sweep.version === version ? (
          <LineChart
            title="Accuracy on 500 test digits against FGSM strength epsilon"
            series={[{ label: 'accuracy', color: 'var(--series-green)', values: sweep.values }]}
            yMin={0}
            yMax={1}
            formatY={v => `${Math.round(v * 100)}%`}
            xTickLabel={i => `ε ${SWEEP[i].toFixed(2)}`}
            markerX={Math.round(epsilon / 0.025)}
            height={180}
          />
        ) : (
          <button type="button" className="btn" onClick={runSweep}>📉 Measure accuracy for ε = 0 … 0.3 ({SWEEP_IMAGES} test digits)</button>
        )}
      </div>
    </>
  );
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
