import { useEffect, useMemo, useRef, useState } from 'react';
import ganCode from './gan.ts?raw';
import pythonCode from './gan.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { sampleToy, type ToyShape } from '../diffusion/diffusion';
import { smooth } from '../shared/train';
import { Gan, MIXTURE_CENTRES, modesCovered, sampleMixture, type GanLoss } from './gan';
import '../shared/dl-shared.css';
import './gan.css';

const CANVAS = 360;
const RES = 48;
const LO = -2, HI = 2;
const BATCH = 128;
const STEPS_PER_FRAME = 5;
const SHOWN = 600;

type Target = 'mixture' | ToyShape;
const TARGETS: Record<Target, string> = { mixture: 'Ring of 8 Gaussians', moons: 'Two moons', rings: 'Two rings', spiral: 'Spiral' };
const sampleTarget = (t: Target, n: number, random: () => number) => (t === 'mixture' ? sampleMixture(n, random) : sampleToy(t, n, random));

export function GanSection() {
  const [target, setTarget] = useState<Target>('mixture');
  const [loss, setLoss] = useState<GanLoss>('non-saturating');
  const [dSteps, setDSteps] = useState(1);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<{ d: number[]; g: number[] }>({ d: [], g: [] });
  const [frame, setFrame] = useState<{ fake: Float32Array; map: Float32Array } | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [version, setVersion] = useState(0);
  const ganRef = useRef<Gan | null>(null);
  const randomRef = useRef(seededRandom(1));
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const real = useMemo(() => sampleTarget(target, SHOWN, seededRandom(7)), [target]);

  const snapshot = (gan: Gan) => setFrame({ fake: gan.sample(SHOWN, randomRef.current), map: gan.decisionMap(RES, LO, HI) });

  /** A fresh game: new networks, cleared history. */
  const restart = () => {
    setRunning(false);
    setStep(0);
    setHistory({ d: [], g: [] });
    setVersion(v => v + 1);
  };

  // Every new version (and every new target) builds new networks, freed when replaced.
  useEffect(() => {
    const gan = new Gan();
    ganRef.current = gan;
    randomRef.current = seededRandom(1);
    setFrame({ fake: gan.sample(SHOWN, randomRef.current), map: gan.decisionMap(RES, LO, HI) });
    return () => {
      gan.dispose();
      if (ganRef.current === gan) ganRef.current = null;
    };
  }, [version, target]);

  useEffect(() => {
    if (!running || !ganRef.current) return;
    const gan = ganRef.current;
    const id = requestAnimationFrame(() => {
      const d: number[] = [], g: number[] = [];
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        const l = gan.step(sampleTarget(target, BATCH, randomRef.current), randomRef.current, { loss, dSteps });
        d.push(l.d); g.push(l.g);
      }
      setHistory(h => ({ d: [...h.d, ...d].slice(-3000), g: [...h.g, ...g].slice(-3000) }));
      setStep(s => s + STEPS_PER_FRAME);
      snapshot(gan);
    });
    return () => cancelAnimationFrame(id);
  }, [running, step, target, loss, dSteps]);

  // Decision map, real points (gray) and generated points (pink), on a dark plot.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !frame) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS * dpr;
    canvas.height = CANVAS * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cell = CANVAS / RES;
    for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
      const p = frame.map[j * RES + i];
      ctx.fillStyle = `rgb(${15 + 40 * p}, ${18 + 70 * p}, ${30 + 110 * p})`;
      ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
    }
    const toPx = (x: number, y: number): [number, number] => [((x - LO) / (HI - LO)) * CANVAS, ((HI - y) / (HI - LO)) * CANVAS];
    const dots = (pts: Float32Array, colour: string, size: number) => {
      ctx.fillStyle = colour;
      for (let i = 0; i < pts.length / 2; i++) {
        const [x, y] = toPx(pts[2 * i], pts[2 * i + 1]);
        ctx.fillRect(x - size / 2, y - size / 2, size, size);
      }
    };
    dots(real, 'rgba(220, 225, 235, 0.55)', 2.5);
    dots(frame.fake, 'rgba(244, 114, 182, 0.9)', 3);
  }, [frame, real]);

  const covered = frame && target === 'mixture' ? modesCovered(frame.fake, MIXTURE_CENTRES) : null;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {(Object.keys(TARGETS) as Target[]).map(t => (
          <button key={t} type="button" aria-pressed={target === t} className={`btn btn-sm${target === t ? ' btn-primary' : ''}`}
            onClick={() => { setTarget(t); restart(); }}>{TARGETS[t]}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={ganCode} title="gan/gan.ts" python={{ code: pythonCode, title: 'gan/gan.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={() => setRunning(r => !r)}>{running ? '⏸ Pause' : '▶ Train'}</button>
        <button type="button" className="btn btn-sm" onClick={restart}>↺ Reset</button>
        <span className="demo-label">Generator loss:</span>
        <button type="button" aria-pressed={loss === 'non-saturating'} className={`btn btn-sm${loss === 'non-saturating' ? ' btn-primary' : ''}`}
          onClick={() => setLoss('non-saturating')}>−log D(G(z))</button>
        <button type="button" aria-pressed={loss === 'minimax'} className={`btn btn-sm${loss === 'minimax' ? ' btn-primary' : ''}`}
          onClick={() => setLoss('minimax')}>log(1 − D(G(z))) (minimax)</button>
        <label className="dl-slider">D steps per G step <strong>{dSteps}</strong>
          <input type="range" min={1} max={5} value={dSteps} onChange={e => setDSteps(Number(e.target.value))} />
        </label>
        <span className="dl-status ml-auto">step {step}{covered !== null ? ` · modes covered: ${covered}/8` : ''}</span>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Generated (pink) vs real (gray)</h3>
          <canvas ref={canvasRef} className="gan-canvas" style={{ width: CANVAS, maxWidth: '100%', aspectRatio: '1' }} role="img"
            aria-label="Real and generated points over the discriminator's decision map" />
          <p className="caption">Background: the discriminator's D(x), brighter = "looks real"</p>
        </div>
        <div className="demo-panel gan-chart">
          <h3>Losses</h3>
          <LineChart title="Discriminator and generator losses, averaged over 25 steps"
            series={[
              { label: 'discriminator', color: 'var(--series-blue)', values: smooth(history.d, 25) },
              { label: 'generator', color: 'var(--series-pink)', values: smooth(history.g, 25) },
            ]} xLabel="step / 25" height={220} />
          <p className="caption">At the ideal balance D(x) = ½ everywhere: the discriminator loss is 2·ln 2 ≈ 1.39 and the
          non-saturating generator loss ln 2 ≈ 0.69. The losses do not go to zero: they oscillate as each player adapts.</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Two networks, one game</h4>
        <p>The <strong>generator</strong> turns 2-D Gaussian noise into points; the <strong>discriminator</strong> outputs the
        probability that a point came from the data. The discriminator is trained to tell the two apart, the generator to
        make it fail. The background shows the discriminator's current opinion: the generator's points slide uphill on it,
        towards where it says "real", while the discriminator keeps redrawing the boundary.</p>
        <p>The original paper wrote the game as a minimax, with the generator minimising log(1 − D(G(z))). Early in training
        the discriminator easily rejects generated points, and there that loss is flat: its gradient is D(G(z)) ≈ 0. Switch to
        the minimax loss after a reset and compare. The <strong>non-saturating</strong> loss −log D(G(z)) has the same goal
        but a large gradient exactly where the generator is worst.</p>
        <p>Watch the <strong>ring of eight Gaussians</strong>: the generator often lands on a few modes, and the discriminator
        chases it from one to the next instead of forcing it to cover all eight. That is <strong>mode collapse</strong>, the
        classic GAN failure. More discriminator steps per generator step, other losses (Wasserstein) and, eventually, diffusion
        models (the Diffusion lesson) were all answers to how fragile this balance is.</p>
      </div>
    </div>
  );
}
