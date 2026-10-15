import { useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import diffusionCode from './diffusion.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { DigitPicker } from '../shared/DigitPicker';
import type { MnistSplit } from '../shared/mnist';
import { floatImage } from '../shared/render';
import { smooth } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { cosineSchedule, linearSchedule, qSample, sampleToy, type ToyShape } from './diffusion';
import { buildDenoiser, sample, trainStep } from './denoiser';
import '../shared/dl-shared.css';
import './diffusion.css';

type Tab = 'forward' | 'reverse';

export function DiffusionSection() {
  const [tab, setTab] = useState<Tab>('forward');
  const [showCode, setShowCode] = useState(false);
  const { data, error } = useMnist(tab === 'forward');
  return (
    <div className="demo-section">
      <div className="tab-bar">
        <div className="tab-list" role="tablist" aria-label="Diffusion views">
          <button type="button" role="tab" aria-selected={tab === 'forward'} className={`tab-btn${tab === 'forward' ? ' tab-active' : ''}`}
            onClick={() => setTab('forward')}>1 · Adding noise (digits)</button>
          <button type="button" role="tab" aria-selected={tab === 'reverse'} className={`tab-btn${tab === 'reverse' ? ' tab-active' : ''}`}
            onClick={() => setTab('reverse')}>2 · Learning to remove it (2-D points)</button>
        </div>
        <ShowCodeButton code={diffusionCode} title="diffusion/diffusion.ts" open={showCode} onOpenChange={setShowCode} />
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {tab === 'forward' && (data ? <Forward test={data.test} /> : !error && <p className="dl-status">Loading MNIST…</p>)}
      {tab === 'reverse' && <Reverse />}
      <div className="demo-explanation">
        <h4>Generating by denoising</h4>
        <p>The <strong>forward process</strong> needs no learning: mix a little Gaussian noise into the image at each of T
        steps. Because Gaussians add up to a Gaussian, step t can be reached in one jump:</p>
        <p className="formula"><code>x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε</code></p>
        <p>By the last step nothing of the image is left; it is pure N(0, 1) noise. The <strong>schedule</strong> (the β's)
        decides how fast information is destroyed: compare the linear schedule, which wipes the digit out early, with the
        cosine one.</p>
        <p>The <strong>reverse process</strong> is learned. A network is shown x_t and t and trained to predict the noise ε
        that was added, an ordinary regression. To generate, start from pure noise and subtract the predicted noise one step
        at a time (adding back a little fresh noise each time). Tab 2 does this for 2-D points, where you can watch the whole
        cloud flow onto the data distribution. Image generators such as Stable Diffusion do the same with a U-Net on images
        (or their compressed latents), conditioned on a text prompt.</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- forward process on digits

const T_IMG = 1000;
const STRIP = [0, 50, 100, 200, 300, 500, 700, 999];

function Forward({ test }: { test: MnistSplit }) {
  const [index, setIndex] = useState(500);
  const [kind, setKind] = useState<'linear' | 'cosine'>('linear');
  const [t, setT] = useState(200);
  const schedule = useMemo(() => (kind === 'linear' ? linearSchedule(T_IMG) : cosineSchedule(T_IMG)), [kind]);

  const x0 = useMemo(() => Float32Array.from(test.images.subarray(index * 784, (index + 1) * 784), v => 2 * v - 1), [test, index]);
  const noise = useMemo(() => { const r = seededRandom(index); return Float32Array.from({ length: 784 }, () => gaussianRandom(r)); }, [index]);
  const show = (x: Float32Array) => floatImage(x.map(v => (v + 1) / 2), 28, 28);

  return (
    <>
      <div className="demo-controls">
        <DigitPicker split={test} value={index} onChange={setIndex} />
      </div>
      <div className="demo-controls">
        <span className="demo-label">Schedule:</span>
        {(['linear', 'cosine'] as const).map(k => (
          <button key={k} type="button" aria-pressed={kind === k} className={`btn btn-sm${kind === k ? ' btn-primary' : ''}`} onClick={() => setKind(k)}>{k}</button>
        ))}
        <label className="dl-slider">Step t <strong>{t}</strong> / {T_IMG}
          <input type="range" min={0} max={T_IMG - 1} value={t} onChange={e => setT(Number(e.target.value))} className="diff-t" />
        </label>
      </div>
      <div className="dl-grid diff-forward">
        <div className="demo-panel">
          <h3>x_t at t = {t}</h3>
          <PixelCanvas image={show(qSample(x0, t, schedule, noise))} size={224} label={`Digit after ${t} noising steps`} />
          <p className="caption mono">√ᾱ = {Math.sqrt(schedule.alphaBars[t]).toFixed(3)} signal · √(1−ᾱ) = {Math.sqrt(1 - schedule.alphaBars[t]).toFixed(3)} noise</p>
        </div>
        <div className="demo-panel">
          <h3>How much of x₀ is left</h3>
          <LineChart
            title="Signal and noise coefficients across the diffusion steps"
            series={[
              { label: '√ᾱ_t (signal)', color: '#34d399', values: Array.from(schedule.alphaBars, a => Math.sqrt(a)) },
              { label: '√(1 − ᾱ_t) (noise)', color: '#f87171', values: Array.from(schedule.alphaBars, a => Math.sqrt(1 - a)) },
            ]}
            yMin={0}
            yMax={1}
            markerX={t}
            xLabel="t"
          />
        </div>
      </div>
      <div className="demo-panel diff-strip">
        {STRIP.map(s => (
          <button key={s} type="button" className="diff-strip-item" onClick={() => setT(s)} aria-label={`Show step ${s}`}>
            <PixelCanvas image={show(qSample(x0, s, schedule, noise))} size={84} label="" />
            <span className="caption mono">t = {s}</span>
          </button>
        ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- reverse process on 2-D points

const T_TOY = 100;
const TRAIN_STEPS = 3000;
const BATCH = 256;
const SAMPLES = 800;
const VIEW = 2.4;
const CANVAS = 380;

function Reverse() {
  const [shape, setShape] = useState<ToyShape>('spiral');
  const [losses, setLosses] = useState<number[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [frames, setFrames] = useState<Float32Array[] | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [forwardT, setForwardT] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const schedule = useMemo(() => cosineSchedule(T_TOY), []);
  const data = useMemo(() => sampleToy(shape, 2000, seededRandom(1)), [shape]);

  const dispose = () => { abortRef.current?.abort(); modelRef.current?.dispose(); modelRef.current = null; };
  useEffect(() => dispose, []);

  const reset = (next: ToyShape = shape) => {
    dispose();
    setShape(next);
    setLosses([]);
    setFrames(null);
    setPlaying(false);
  };

  const train = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    modelRef.current ??= buildDenoiser();
    const model = modelRef.current;
    const random = seededRandom(losses.length + 7);
    const batchLosses: number[] = [];
    try {
      for (let s = 0; s < TRAIN_STEPS && !controller.signal.aborted; s++) {
        batchLosses.push(await trainStep(model, data, schedule, BATCH, random));
        if (s % 25 === 24) {
          setProgress(s + 1);
          setLosses(l => [...l, ...batchLosses.splice(0)]);
          await tf.nextFrame();
        }
      }
      if (!controller.signal.aborted) generate();
    } catch (err) {
      // Reset or leaving the page disposes the model mid-training: that is a normal stop.
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      if (abortRef.current === controller) setProgress(null);
    }
  };

  const generate = () => {
    const model = modelRef.current;
    if (!model) return;
    setFrames(sample(model, schedule, SAMPLES, seededRandom(Date.now() % 1e6)));
    setFrame(0);
    setPlaying(true);
  };

  useEffect(() => {
    if (!playing || !frames) return;
    if (frame >= frames.length - 1) { const id = setTimeout(() => setPlaying(false), 0); return () => clearTimeout(id); }
    const id = setTimeout(() => setFrame(f => f + 1), 40);
    return () => clearTimeout(id);
  }, [playing, frame, frames]);

  const forwardPoints = useMemo(() => {
    if (forwardT === 0) return data.subarray(0, SAMPLES * 2);
    const r = seededRandom(3);
    return qSample(data.subarray(0, SAMPLES * 2), forwardT - 1, schedule, Float32Array.from({ length: SAMPLES * 2 }, () => gaussianRandom(r)));
  }, [data, forwardT, schedule]);

  const t = frames ? T_TOY - frame : forwardT;

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {(['spiral', 'moons', 'rings'] as const).map(s => (
          <button key={s} type="button" aria-pressed={shape === s} className={`btn btn-sm${shape === s ? ' btn-primary' : ''}`}
            disabled={progress !== null} onClick={() => reset(s)}>{s}</button>
        ))}
        {progress !== null ? (
          <>
            <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>⏹ Stop</button>
            <span className="dl-status">step {progress} / {TRAIN_STEPS}</span>
          </>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" onClick={train}>▶ Train the denoiser ({TRAIN_STEPS} steps)</button>
        )}
        <button type="button" className="btn btn-sm" disabled={!modelRef.current || progress !== null} onClick={generate}>✨ Generate {SAMPLES} points</button>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      <div className="dl-grid diff-reverse">
        <div className="demo-panel">
          <h3>{frames ? `Reverse process: t = ${t}` : forwardT ? `Forward process: t = ${forwardT}` : 'Training data'}</h3>
          <PointCloud points={frames ? frames[frame] : forwardPoints} reference={data} />
          {frames ? (
            <div className="dl-sliders diff-frame">
              <button type="button" className="btn btn-sm" onClick={() => { if (frame >= frames.length - 1) setFrame(0); setPlaying(p => !p); }}>
                {playing ? '⏸' : '▶'}
              </button>
              <label className="dl-slider">Denoising step <strong>{frame}</strong> / {T_TOY}
                <input type="range" min={0} max={T_TOY} value={frame} onChange={e => { setPlaying(false); setFrame(Number(e.target.value)); }} />
              </label>
              <button type="button" className="btn btn-sm" onClick={() => setFrames(null)}>Show forward process</button>
            </div>
          ) : (
            <label className="dl-slider diff-frame">Noising step <strong>{forwardT}</strong> / {T_TOY}
              <input type="range" min={0} max={T_TOY} value={forwardT} onChange={e => setForwardT(Number(e.target.value))} />
            </label>
          )}
        </div>
        <div className="demo-panel">
          <h3>Denoiser training loss (‖ε − ε̂‖²)</h3>
          <LineChart title="Noise-prediction loss, averaged over 50 steps" series={[{ label: 'loss', color: '#f87171', values: smooth(losses, 50) }]}
            xLabel="× 50 steps" yMin={0} />
          <p className="caption">A loss of 1 means the network has learned nothing (predicting 0 noise); below 1 it is recovering structure.</p>
        </div>
      </div>
    </>
  );
}

function PointCloud({ points, reference }: { points: Float32Array; reference: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS * dpr;
    canvas.height = CANVAS * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1018';
    ctx.fillRect(0, 0, CANVAS, CANVAS);
    const px = (v: number) => ((v + VIEW) / (2 * VIEW)) * CANVAS;
    const py = (v: number) => ((VIEW - v) / (2 * VIEW)) * CANVAS;
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    for (let i = 0; i < reference.length / 2; i++) ctx.fillRect(px(reference[2 * i]) - 1, py(reference[2 * i + 1]) - 1, 2, 2);
    ctx.fillStyle = '#38bdf8';
    for (let i = 0; i < points.length / 2; i++) ctx.fillRect(px(points[2 * i]) - 1.5, py(points[2 * i + 1]) - 1.5, 3, 3);
  }, [points, reference]);
  return <canvas ref={ref} className="diff-cloud" style={{ width: CANVAS, maxWidth: '100%', aspectRatio: '1' }} role="img"
    aria-label="Point cloud: the current samples in blue over the training data in grey" />;
}
