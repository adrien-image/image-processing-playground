import { useEffect, useMemo, useState } from 'react';
import learnCode from './learnKernel.ts?raw';
import pythonCode from './learn_kernel.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData } from '../../hooks/useImageData';
import { scene } from '../../lib/demoImages';
import { divergingColor } from '../../lib/colormap';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { convolve } from '../../image-processing/processors/utils';
import { gaussianKernel, LAPLACIAN_3x3, SOBEL_X, SOBEL_Y, sharpenKernel } from '../../image-processing/processors/kernels';
import { floatImage, heatImage, maxAbs, signedImage } from '../shared/render';
import { descend, evaluateKernel, kernelDistance, randomKernel, zeroKernel, type Kernel, type OptimizerState } from './learnKernel';
import '../shared/dl-shared.css';
import './learn-kernel.css';

const SIZE = 64;
const DISPLAY = 200;
const STEPS_PER_FRAME = 2;
const MAX_HISTORY = 4000;

const round1 = (k: Kernel) => k.map(r => r.map(v => Math.round(v * 10) / 10));

const TARGETS = {
  'sobel-x': { label: 'Sobel X', kernel: SOBEL_X },
  'sobel-y': { label: 'Sobel Y', kernel: SOBEL_Y },
  blur: { label: 'Gaussian blur', kernel: gaussianKernel(1, 1) },
  sharpen: { label: 'Sharpen', kernel: sharpenKernel(1) },
  laplacian: { label: 'Laplacian', kernel: LAPLACIAN_3x3 },
  mystery: { label: 'Mystery', kernel: round1(randomKernel(3, seededRandom(42), 1)) },
} as const;
type TargetKey = keyof typeof TARGETS;

type Init = 'zeros' | 'random';
const initialState = (init: Init, seed: number): OptimizerState => ({
  kernel: init === 'zeros' ? zeroKernel(3) : randomKernel(3, seededRandom(seed)),
  velocity: zeroKernel(3),
});

export function LearnKernelSection() {
  const [source] = useState(() => scene(SIZE));
  const { image } = useImageData(source, SIZE);
  const [inputKind, setInputKind] = useState<'scene' | 'noise'>('scene');
  const input = useMemo(() => {
    if (inputKind === 'noise') {
      const random = seededRandom(5);
      return Float64Array.from({ length: SIZE * SIZE }, () => random());
    }
    return image ? image.gray.map(v => v / 255) : null;
  }, [image, inputKind]);

  const [targetKey, setTargetKey] = useState<TargetKey>('sobel-x');
  const [noise, setNoise] = useState(0);
  const [logLr, setLogLr] = useState(Math.log10(0.1));
  const [momentum, setMomentum] = useState(true);
  const [init, setInit] = useState<Init>('zeros');
  const [seed, setSeed] = useState(1);
  const [opt, setOpt] = useState<OptimizerState>(() => initialState('zeros', 1));
  const [history, setHistory] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const lr = 10 ** logLr;
  const trueKernel = TARGETS[targetKey].kernel;

  const target = useMemo(() => {
    if (!input) return null;
    const clean = convolve(input, SIZE, SIZE, trueKernel as Kernel);
    if (noise === 0) return clean;
    const random = seededRandom(99);
    return clean.map(v => v + noise * gaussianRandom(random));
  }, [input, trueKernel, noise]);

  const fit = useMemo(() => (input && target ? evaluateKernel(input, target, SIZE, SIZE, opt.kernel) : null), [input, target, opt.kernel]);
  const diverged = fit !== null && (!Number.isFinite(fit.loss) || fit.loss > 1e6);

  const reset = (nextInit = init, nextSeed = seed) => {
    setRunning(false);
    setOpt(initialState(nextInit, nextSeed));
    setHistory([]);
  };

  const step = (count: number) => {
    if (!input || !target) return;
    const { state, losses } = runSteps(opt, input, target, count, lr, momentum ? 0.9 : 0);
    setOpt(state);
    setHistory(h => [...h, ...losses].slice(-MAX_HISTORY));
  };

  // Training loop: a few steps per animation frame while running (and not diverged).
  const training = running && !diverged;
  useEffect(() => {
    if (!training || !input || !target) return;
    const id = requestAnimationFrame(() => {
      const { state, losses } = runSteps(opt, input, target, STEPS_PER_FRAME, lr, momentum ? 0.9 : 0);
      setOpt(state);
      setHistory(h => [...h, ...losses].slice(-MAX_HISTORY));
    });
    return () => cancelAnimationFrame(id);
  }, [training, opt, input, target, lr, momentum]);

  const signed = Math.abs((trueKernel as Kernel).flat().reduce((a, b) => a + b, 0)) < 1e-9;
  const scale = target ? maxAbs(target) : 1;
  const show = (values: Float64Array) => (signed ? signedImage(values, SIZE, SIZE, scale) : floatImage(values, SIZE, SIZE));
  const error = fit && target ? fit.prediction.map((p, i) => Math.abs(p - target[i])) : null;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Hidden kernel:</span>
        {(Object.keys(TARGETS) as TargetKey[]).map(k => (
          <button key={k} type="button" aria-pressed={targetKey === k}
            className={`btn btn-sm${targetKey === k ? ' btn-primary' : ''}`}
            onClick={() => { setTargetKey(k); reset(); }}>
            {TARGETS[k].label}
          </button>
        ))}
        <span className="demo-label lk-gap">Training image:</span>
        <button type="button" aria-pressed={inputKind === 'scene'} className={`btn btn-sm${inputKind === 'scene' ? ' btn-primary' : ''}`}
          onClick={() => { setInputKind('scene'); reset(); }}>Scene</button>
        <button type="button" aria-pressed={inputKind === 'noise'} className={`btn btn-sm${inputKind === 'noise' ? ' btn-primary' : ''}`}
          onClick={() => { setInputKind('noise'); reset(); }}>Random noise</button>
        <span className="ml-auto">
          <ShowCodeButton code={learnCode} title="learn-kernel/learnKernel.ts" python={{ code: pythonCode, title: 'learn-kernel/learn_kernel.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={() => setRunning(r => !r)} disabled={diverged}>
          {training ? '⏸ Pause' : '▶ Train'}
        </button>
        <button type="button" className="btn btn-sm" onClick={() => step(1)} disabled={training || diverged}>Step</button>
        <button type="button" className="btn btn-sm" onClick={() => step(100)} disabled={training || diverged}>+100 steps</button>
        <button type="button" className="btn btn-sm" onClick={() => reset()}>↺ Reset</button>
        <div className="dl-sliders">
          <label className="dl-slider">
            Learning rate η <strong>{lr < 0.01 ? lr.toExponential(1) : lr.toFixed(lr < 0.1 ? 3 : 2)}</strong>
            <input type="range" min={-3} max={0} step={0.05} value={logLr} onChange={e => setLogLr(Number(e.target.value))} />
          </label>
          <label className="dl-slider">
            <input type="checkbox" checked={momentum} onChange={e => setMomentum(e.target.checked)} /> Momentum 0.9
          </label>
          <label className="dl-slider">
            Label noise σ <strong>{noise.toFixed(2)}</strong>
            <input type="range" min={0} max={0.5} step={0.05} value={noise} onChange={e => { setNoise(Number(e.target.value)); reset(); }} />
          </label>
          <label className="dl-slider">
            Start from
            <select className="select lk-select" value={init} onChange={e => { const v = e.target.value as Init; setInit(v); reset(v); }}>
              <option value="zeros">zeros</option>
              <option value="random">random weights</option>
            </select>
          </label>
          {init === 'random' && (
            <button type="button" className="btn btn-sm" onClick={() => { setSeed(seed + 1); reset(init, seed + 1); }}>🎲 New start</button>
          )}
        </div>
      </div>

      {diverged && (
        <div className="error-banner" role="alert">
          ⚠ The loss exploded: the learning rate is too high, so every step overshoots the minimum by more than the last.
          Lower η and press Reset.
        </div>
      )}

      <div className="dl-grid">
        <div className="demo-panel">
          <h3>Input x</h3>
          <PixelCanvas image={input ? floatImage(input, SIZE, SIZE) : null} size={DISPLAY} label="Input image" />
        </div>
        <div className="demo-panel">
          <h3>Target t = K* ⋆ x</h3>
          <PixelCanvas image={target ? show(target) : null} size={DISPLAY} label="Target image made with the hidden kernel" />
        </div>
        <div className="demo-panel">
          <h3>Prediction K ⋆ x</h3>
          <PixelCanvas image={fit ? show(fit.prediction) : null} size={DISPLAY} label="Output of the learned kernel" />
        </div>
        <div className="demo-panel">
          <h3>Error |K ⋆ x − t|</h3>
          <PixelCanvas image={error ? heatImage(error, SIZE, SIZE, scale) : null} size={DISPLAY} label="Absolute error" />
        </div>
      </div>

      <div className="dl-grid lk-bottom">
        <div className="demo-panel">
          <h3>Kernels</h3>
          <div className="lk-kernels">
            <KernelGrid kernel={opt.kernel} title="Learned K" />
            <KernelGrid kernel={trueKernel as Kernel} title={targetKey === 'mystery' ? 'Hidden K* (mystery)' : 'Hidden K*'} />
            {fit && <KernelGrid kernel={fit.gradient} title="Gradient ∂L/∂K" />}
          </div>
          <div className="dl-stat-row">
            <span>Step <strong>{history.length}</strong></span>
            <span>Loss <strong>{fit ? fit.loss.toExponential(2) : '—'}</strong></span>
            <span>‖K − K*‖ <strong>{kernelDistance(opt.kernel, trueKernel as Kernel).toFixed(3)}</strong></span>
          </div>
        </div>
        <div className="demo-panel">
          <h3>Loss (mean squared error)</h3>
          <LineChart
            title="Training loss per step, logarithmic scale"
            series={[{ label: 'loss', color: '#f87171', values: history }]}
            logY
            xLabel="step"
          />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>A CNN learns its kernels exactly like this</h4>
        <p>The Convolution lesson uses hand-designed kernels. A convolutional network starts from zeros or random numbers
        instead and <strong>learns</strong> the weights: it compares its output with the answer it should have given and nudges
        every weight downhill on the error.</p>
        <p className="formula"><code>K ← K − η · ∂L/∂K,&nbsp;&nbsp; ∂L/∂K[i,j] = 2/N · Σ (K⋆x − t) · x shifted by (i, j)</code></p>
        <p>Each weight's gradient is the error image multiplied by the input shifted under that weight: a weight that sits on
        pixels correlated with the error gets pushed. Try a <strong>learning rate</strong> above ~0.4 to see training diverge,
        turn off <strong>momentum</strong> to see how much slower plain gradient descent is, and add <strong>label noise</strong>:
        the kernel is still recovered almost exactly, because the noise averages out over 4,096 pixels.</p>
        <p>Notice that on the <strong>Scene</strong> the loss soon becomes tiny while ‖K − K*‖ stays large: the picture is
        mostly smooth, so several different kernels give almost the same output and the gradient barely tells them apart
        (the problem is <em>ill-conditioned</em>). Train on <strong>Random noise</strong>, where every pixel is independent,
        and the weights converge exactly. Varied training data matters as much as the optimiser. A real CNN layer does all
        this for dozens of kernels at once, with the target coming from the layers above instead of a known filter.</p>
      </div>
    </div>
  );
}

/** `count` gradient-descent steps; stops early once the loss blows up. */
function runSteps(start: OptimizerState, input: Float64Array, target: Float64Array, count: number, lr: number, momentum: number) {
  let state = start;
  const losses: number[] = [];
  for (let i = 0; i < count; i++) {
    const { loss, gradient } = evaluateKernel(input, target, SIZE, SIZE, state.kernel);
    losses.push(loss);
    if (!Number.isFinite(loss) || loss > 1e6) break;
    state = descend(state, gradient, lr, momentum);
  }
  return { state, losses };
}

function KernelGrid({ kernel, title }: { kernel: Kernel; title: string }) {
  const scale = maxAbs(kernel.flat()) || 1;
  return (
    <figure className="lk-kernel">
      <figcaption>{title}</figcaption>
      <div className="lk-kernel-grid" style={{ gridTemplateColumns: `repeat(${kernel.length}, 1fr)` }}>
        {kernel.flat().map((v, i) => {
          const [r, g, b] = divergingColor(v / scale);
          return (
            <span key={i} className="lk-cell" style={{ background: `rgb(${r * 0.6}, ${g * 0.6}, ${b * 0.6})` }}>
              {Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toExponential(0) : v.toFixed(2)) : '∞'}
            </span>
          );
        })}
      </div>
    </figure>
  );
}
