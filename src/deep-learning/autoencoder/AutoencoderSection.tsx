import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import * as tf from '@tensorflow/tfjs';
import vaeCode from './vae.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { seededRandom } from '../../lib/random';
import { toImageData } from '../../lib/canvas';
import type { MnistData } from '../shared/mnist';
import { CLASS_COLORS, floatImage } from '../shared/render';
import { gatherRows, optimise, shuffled } from '../shared/train';
import { useMnist } from '../shared/useMnist';
import { Autoencoder } from './vae';
import '../shared/dl-shared.css';
import './autoencoder.css';

const BATCH = 50;
const SCATTER = 360;
const SHOWN = 1000;
const GRID = 15;

type Mode = 'vae' | 'ae';
interface EpochLog { reconstruction: number; kl: number }

export function AutoencoderSection() {
  const { data, error } = useMnist();
  const [showCode, setShowCode] = useState(false);
  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Compress every digit to just two numbers, then decode any point of the plane</span>
        <span className="ml-auto">
          <ShowCodeButton code={vaeCode} title="autoencoder/vae.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {data ? <Trainer data={data} /> : !error && <p className="dl-status">Loading MNIST…</p>}
      <div className="demo-explanation">
        <h4>Autoencoders and latent spaces</h4>
        <p>An <strong>autoencoder</strong> is trained to output its own input, through a bottleneck: here, 2 numbers for 784
        pixels. To reconstruct well it must put similar digits close together, so the 2-D codes form clusters by digit
        without ever being told the labels. That plane is the <strong>latent space</strong>.</p>
        <p>A plain autoencoder is free to scatter the clusters anywhere, leaving empty gaps: points in a gap decode to
        nonsense. A <strong>variational</strong> autoencoder encodes each digit as a small Gaussian cloud and is penalised
        (KL divergence) for straying from a unit Gaussian. The clusters pack around the origin, every point decodes to
        something digit-like, and you can generate new digits by picking points. Compare the decoded grid in both modes, and
        raise β to trade sharper reconstructions for a tidier latent space.</p>
        <p className="formula"><code>loss = BCE(x, decode(z)) + β · KL(N(μ, σ²) ‖ N(0, 1)),&nbsp;&nbsp; z = μ + σ · ε</code></p>
      </div>
    </div>
  );
}

function Trainer({ data }: { data: MnistData }) {
  const [mode, setMode] = useState<Mode>('vae');
  const [beta, setBeta] = useState(1);
  const [epochsToRun, setEpochsToRun] = useState(15);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<Autoencoder | null>(null);
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

  const reset = (next: Mode = mode) => {
    dispose();
    setMode(next);
    setHistory([]);
    setVersion(v => v + 1);
  };

  const train = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    modelRef.current ??= new Autoencoder();
    optRef.current ??= tf.train.adam(0.002);
    const model = modelRef.current, optimizer = optRef.current;
    const variational = mode === 'vae';
    const random = seededRandom(history.length + 1);
    const steps = Math.floor(data.train.count / BATCH);
    const val = tf.tensor2d(data.val.images, [data.val.count, 784]);
    try {
      for (let e = 0; e < epochsToRun && !controller.signal.aborted; e++) {
        const order = shuffled(data.train.count, random);
        await optimise(optimizer, steps,
          s => model.loss(tf.tensor2d(gatherRows(data.train.images, 784, order.slice(s * BATCH, (s + 1) * BATCH)), [BATCH, 784]), variational, beta).total,
          { signal: controller.signal, varList: model.variables, onStep: s => { if (s % 10 === 0) setProgress(`epoch ${e + 1}/${epochsToRun}, batch ${s + 1}/${steps}`); } });
        if (controller.signal.aborted || modelRef.current !== model) break;
        const parts = tf.tidy(() => {
          const l = model.loss(val, variational, beta);
          return { reconstruction: l.reconstruction.dataSync()[0], kl: l.kl.dataSync()[0] };
        });
        setHistory(h => [...h, parts]);
        setVersion(v => v + 1);
      }
    } catch (err) {
      // Reset or leaving the page disposes the model under a running epoch: that is a normal stop.
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
      val.dispose();
      if (abortRef.current === controller) setProgress(null);
    }
  };

  const view = useMemo(() => {
    const model = modelRef.current;
    if (!model || version < 0) return null;
    const codes = model.codes(data.test.images, SHOWN);
    let lo = Infinity, hi = -Infinity;
    for (const v of codes) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.05 || 1;
    const range: [number, number] = mode === 'vae' ? [-3.5, 3.5] : [lo - pad, hi + pad];
    // A grid of decoded points across the visible range.
    const points = new Float32Array(GRID * GRID * 2);
    for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
      points[(r * GRID + c) * 2] = range[0] + ((c + 0.5) / GRID) * (range[1] - range[0]);
      points[(r * GRID + c) * 2 + 1] = range[1] - ((r + 0.5) / GRID) * (range[1] - range[0]);
    }
    const decoded = model.decode(points);
    const rgba = new Uint8ClampedArray(GRID * 28 * GRID * 28 * 4);
    for (let k = 0; k < GRID * GRID; k++) {
      const gr = Math.floor(k / GRID), gc = k % GRID;
      for (let p = 0; p < 784; p++) {
        const y = gr * 28 + Math.floor(p / 28), x = gc * 28 + (p % 28);
        const i = (y * GRID * 28 + x) * 4;
        rgba[i] = rgba[i + 1] = rgba[i + 2] = decoded[k * 784 + p] * 255;
        rgba[i + 3] = 255;
      }
    }
    const examples = firstOfEachDigit(data.test.labels);
    const recon = model.reconstruct(gatherRows(data.test.images, 784, examples), examples.length);
    return { codes, range, manifold: toImageData(rgba, GRID * 28, GRID * 28), recon, examples };
  }, [data, version, mode]);

  const hovered = useMemo(() => {
    const model = modelRef.current;
    if (!model || !hover || version < 0) return null;
    return floatImage(model.decode(Float32Array.from(hover)), 28, 28);
  }, [hover, version]);

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">Model:</span>
        <button type="button" aria-pressed={mode === 'vae'} className={`btn btn-sm${mode === 'vae' ? ' btn-primary' : ''}`} disabled={!!progress}
          onClick={() => reset('vae')}>Variational (VAE)</button>
        <button type="button" aria-pressed={mode === 'ae'} className={`btn btn-sm${mode === 'ae' ? ' btn-primary' : ''}`} disabled={!!progress}
          onClick={() => reset('ae')}>Plain autoencoder</button>
        {mode === 'vae' && (
          <label className="dl-slider">β <strong>{beta}</strong>
            <input type="range" min={0.25} max={4} step={0.25} value={beta} disabled={!!progress} onChange={e => setBeta(Number(e.target.value))} />
          </label>
        )}
        <label className="dl-slider">Epochs <strong>{epochsToRun}</strong>
          <input type="range" min={1} max={30} value={epochsToRun} disabled={!!progress} onChange={e => setEpochsToRun(Number(e.target.value))} />
        </label>
        {progress ? (
          <button type="button" className="btn btn-sm" onClick={() => abortRef.current?.abort()}>⏹ Stop</button>
        ) : (
          <button type="button" className="btn btn-sm btn-primary" onClick={train}>▶ Train {history.length ? 'more' : ''}</button>
        )}
        <button type="button" className="btn btn-sm" disabled={!!progress} onClick={() => reset()}>↺ Reset</button>
        {progress && <span className="dl-status">{progress}</span>}
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {!view && <p className="muted">Press Train. Each epoch passes the 6,000 training digits through the network once.</p>}
      {view && (
        <div className="ae-layout">
          <div className="demo-panel">
            <h3>Latent space: {SHOWN} test digits</h3>
            <Scatter codes={view.codes} labels={data.test.labels} range={view.range} hover={hover} onHover={setHover} />
            <div className="ae-legend">
              {CLASS_COLORS.map((c, d) => <span key={d}><i style={{ background: c }} />{d}</span>)}
            </div>
          </div>
          <div className="demo-panel ae-preview">
            <h3>Decoded point</h3>
            <PixelCanvas image={hovered} size={168} label="Image decoded from the point under the pointer" placeholder="Hover the plane" />
            <p className="caption mono">{hover ? `z = (${hover[0].toFixed(2)}, ${hover[1].toFixed(2)})` : 'move over the latent space'}</p>
            <h3>Reconstructions</h3>
            <div className="ae-recon">
              {view.examples.map((index, k) => (
                <div key={k}>
                  <PixelCanvas image={floatImage(data.test.images.subarray(index * 784, (index + 1) * 784), 28, 28)} size={40} label="Original" />
                  <PixelCanvas image={floatImage(view.recon.subarray(k * 784, (k + 1) * 784), 28, 28)} size={40} label="Reconstruction" />
                </div>
              ))}
            </div>
            <p className="caption">top: input · bottom: after the 2-number bottleneck</p>
          </div>
          <div className="demo-panel">
            <h3>Decoding a {GRID}×{GRID} grid of the plane</h3>
            <PixelCanvas image={view.manifold} size={SCATTER} label="Images decoded from a regular grid of latent points" />
          </div>
          <div className="demo-panel">
            <h3>Validation loss per epoch</h3>
            <LineChart
              title="Reconstruction and KL loss per epoch"
              series={[
                { label: 'reconstruction (BCE)', color: '#60a5fa', values: history.map(h => h.reconstruction) },
                ...(mode === 'vae' ? [{ label: 'KL', color: '#f472b6', values: history.map(h => h.kl) }] : []),
              ]}
              xStart={1}
              xLabel="epoch"
              yMin={0}
            />
          </div>
        </div>
      )}
    </>
  );
}

/** Index of the first test image of each digit 0–9. */
function firstOfEachDigit(labels: Uint8Array): number[] {
  return Array.from({ length: 10 }, (_, d) => labels.indexOf(d)).filter(i => i >= 0);
}

interface ScatterProps {
  codes: Float32Array;
  labels: Uint8Array;
  range: [number, number];
  hover: [number, number] | null;
  onHover: (z: [number, number] | null) => void;
}

function Scatter({ codes, labels, range, hover, onHover }: ScatterProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [lo, hi] = range;
  const toPx = (v: number, flip: boolean) => ((flip ? hi - v : v - lo) / (hi - lo)) * SCATTER;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = SCATTER * dpr;
    canvas.height = SCATTER * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1018';
    ctx.fillRect(0, 0, SCATTER, SCATTER);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath();
    ctx.moveTo(toPx(0, false), 0); ctx.lineTo(toPx(0, false), SCATTER);
    ctx.moveTo(0, toPx(0, true)); ctx.lineTo(SCATTER, toPx(0, true));
    ctx.stroke();
    for (let i = 0; i < codes.length / 2; i++) {
      ctx.fillStyle = CLASS_COLORS[labels[i]];
      ctx.globalAlpha = 0.75;
      ctx.fillRect(toPx(codes[2 * i], false) - 1.5, toPx(codes[2 * i + 1], true) - 1.5, 3, 3);
    }
    ctx.globalAlpha = 1;
    if (hover) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(toPx(hover[0], false), toPx(hover[1], true), 7, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  const move = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = lo + ((e.clientX - rect.left) / rect.width) * (hi - lo);
    const y = hi - ((e.clientY - rect.top) / rect.height) * (hi - lo);
    onHover([x, y]);
  };

  return (
    <canvas ref={ref} className="ae-scatter" style={{ width: SCATTER, maxWidth: '100%', aspectRatio: '1' }}
      onPointerMove={move} onPointerDown={move} role="img"
      aria-label="Scatter plot of the 2-D codes of test digits, coloured by digit; hover to decode a point" />
  );
}
