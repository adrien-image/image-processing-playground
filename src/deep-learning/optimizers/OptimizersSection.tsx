import { useEffect, useMemo, useState, type MouseEvent } from 'react';
import optimizersCode from './optimizers.ts?raw';
import pythonCode from './optimizers.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { OPTIMIZER_LABELS, OPTIMIZERS, runOptimizer, SURFACES, type OptimizerName, type Surface, type SurfaceName } from './optimizers';
import '../shared/dl-shared.css';
import './optimizers.css';

const RES = 220;
const DISPLAY = 440;

const COLORS: Record<OptimizerName, string> = {
  sgd: 'var(--series-red)', momentum: 'var(--series-orange)', nesterov: 'var(--series-pink)',
  rmsprop: 'var(--series-blue)', adam: 'var(--series-green)',
};

/** log(loss) as shades of blue with contour bands; the image (and its look) is the same in both themes. */
function surfaceImage(s: Surface): ImageData {
  const { xMin, xMax, yMin, yMax } = s.domain;
  const fMin = s.f(...s.minimum);
  const values = new Float64Array(RES * RES);
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < RES; j++) {
    for (let i = 0; i < RES; i++) {
      const x = xMin + ((i + 0.5) / RES) * (xMax - xMin), y = yMax - ((j + 0.5) / RES) * (yMax - yMin);
      const v = Math.log10(s.f(x, y) - fMin + 1e-3);
      values[j * RES + i] = v;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
  }
  return rgbImage(RES, RES, k => {
    const t = (values[k] - lo) / (hi - lo);
    const band = (t * 14) % 1 < 0.08 ? 0.72 : 1;
    return [band * (20 + 120 * t), band * (40 + 110 * t), band * (90 + 120 * t)];
  });
}

export function OptimizersSection() {
  const [surfaceName, setSurfaceName] = useState<SurfaceName>('ravine');
  const surface: Surface = SURFACES[surfaceName];
  const [start, setStart] = useState<[number, number]>(surface.start);
  const [enabled, setEnabled] = useState<Record<OptimizerName, boolean>>({ sgd: true, momentum: true, nesterov: false, rmsprop: true, adam: true });
  const [rates, setRates] = useState<Record<OptimizerName, number>>(surface.rates);
  const [steps, setSteps] = useState(surface.steps);
  const [noise, setNoise] = useState(0);
  const [shown, setShown] = useState<number | null>(null);
  const [showCode, setShowCode] = useState(false);

  const chooseSurface = (name: SurfaceName) => {
    setSurfaceName(name);
    setStart(SURFACES[name].start);
    setRates(SURFACES[name].rates);
    setSteps(SURFACES[name].steps);
    setShown(null);
  };

  const background = useMemo(() => surfaceImage(surface), [surface]);
  const runs = useMemo(() => Object.fromEntries(OPTIMIZERS.map(name => [
    name, runOptimizer(surface, name, rates[name], steps, start, { noise, random: seededRandom(11) }),
  ])) as Record<OptimizerName, ReturnType<typeof runOptimizer>>, [surface, rates, steps, start, noise]);

  // Playback: reveal the paths step by step.
  useEffect(() => {
    if (shown === null || shown >= steps) return;
    const id = requestAnimationFrame(() => setShown(s => (s === null ? null : Math.min(steps, s + Math.max(1, Math.round(steps / 150))))));
    return () => cancelAnimationFrame(id);
  }, [shown, steps]);
  const upto = shown ?? steps;

  const { xMin, xMax, yMin, yMax } = surface.domain;
  const toSvg = ([x, y]: [number, number]) => [((x - xMin) / (xMax - xMin)) * RES, ((yMax - y) / (yMax - yMin)) * RES];
  const pick = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width, fy = (e.clientY - rect.top) / rect.height;
    setStart([xMin + fx * (xMax - xMin), yMax - fy * (yMax - yMin)]);
    setShown(null);
  };
  const active = OPTIMIZERS.filter(n => enabled[n]);
  const [sx, sy] = toSvg(start), [mx, my] = toSvg(surface.minimum);

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Surface:</span>
        {(Object.keys(SURFACES) as SurfaceName[]).map(name => (
          <button key={name} type="button" aria-pressed={surfaceName === name} className={`btn btn-sm${surfaceName === name ? ' btn-primary' : ''}`}
            onClick={() => chooseSurface(name)}>{SURFACES[name].label}</button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => setShown(0)}>▶ Replay</button>
        <span className="ml-auto">
          <ShowCodeButton code={optimizersCode} title="optimizers/optimizers.ts" python={{ code: pythonCode, title: 'optimizers/optimizers.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Steps <strong>{steps}</strong>
            <input type="range" min={10} max={2000} step={10} value={steps} onChange={e => { setSteps(Number(e.target.value)); setShown(null); }} />
          </label>
          <label className="dl-slider">Gradient noise σ <strong>{noise.toFixed(1)}</strong>
            <input type="range" min={0} max={5} step={0.1} value={noise} onChange={e => setNoise(Number(e.target.value))} />
          </label>
        </div>
      </div>

      <div className="opt-layout">
        <div className="demo-panel">
          <h3>{surface.label} · click to start elsewhere</h3>
          <div className="opt-picker" onClick={pick} role="presentation">
            <PixelCanvas image={background} size={DISPLAY} pixelated={false} label="The loss surface, darker is lower, with each optimizer's path">
              <svg className="opt-overlay" viewBox={`0 0 ${RES} ${RES}`} preserveAspectRatio="none" aria-hidden="true">
                {active.map(name => (
                  <polyline key={name} className="opt-path" style={{ stroke: COLORS[name] }}
                    points={runs[name].path.slice(0, upto + 1).map(p => toSvg(p).join(',')).join(' ')} />
                ))}
                {active.map(name => {
                  const p = runs[name].path[Math.min(upto, runs[name].path.length - 1)];
                  const [x, y] = toSvg(p);
                  return <circle key={name} className="opt-head" cx={x} cy={y} r={3} style={{ fill: COLORS[name] }} />;
                })}
                <circle className="opt-start" cx={sx} cy={sy} r={3.5} />
                <path className="opt-min" d={`M${mx - 4},${my} L${mx + 4},${my} M${mx},${my - 4} L${mx},${my + 4}`} />
              </svg>
            </PixelCanvas>
          </div>
          <p className="caption">○ start · + minimum · contour bands every equal step of log(loss)</p>
        </div>

        <div className="opt-side">
          <table className="opt-table">
            <thead>
              <tr><th scope="col">optimizer</th><th scope="col">learning rate</th><th scope="col">final loss</th></tr>
            </thead>
            <tbody>
              {OPTIMIZERS.map(name => {
                const run = runs[name];
                const loss = run.losses[Math.min(upto, run.losses.length - 1)];
                return (
                  <tr key={name} className={enabled[name] ? '' : 'opt-off'}>
                    <th scope="row">
                      <label className="opt-name">
                        <input type="checkbox" checked={enabled[name]} onChange={e => setEnabled({ ...enabled, [name]: e.target.checked })} />
                        <span className="opt-swatch" style={{ background: COLORS[name] }} />{OPTIMIZER_LABELS[name]}
                      </label>
                    </th>
                    <td>
                      <input type="range" min={-5} max={0} step={0.05} value={Math.log10(rates[name])} aria-label={`${OPTIMIZER_LABELS[name]} learning rate`}
                        onChange={e => setRates({ ...rates, [name]: 10 ** Number(e.target.value) })} />
                      <span className="mono">{rates[name].toPrecision(2)}</span>
                    </td>
                    <td className="mono">{run.diverged && upto >= run.losses.length - 1 ? 'diverged' : loss.toExponential(1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <LineChart title="Loss against step, for each optimizer" logY
            series={active.map(name => ({ label: OPTIMIZER_LABELS[name], color: COLORS[name], values: runs[name].losses.slice(0, upto + 1) }))}
            xLabel="step" height={220} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Same gradient, different steps</h4>
        <p>Every optimizer here sees the same gradient at the same point; what differs is what it remembers. Plain
        <strong> SGD</strong> steps against the gradient: p ← p − η·g. On the <em>ravine</em> the gradient points mostly across the
        narrow valley, so SGD zig-zags, and a learning rate above 2 divided by the steepest curvature (0.5 here) makes it
        diverge: the slow direction then never gets a big enough step.</p>
        <p><strong>Momentum</strong> keeps a velocity b ← μ·b + g: the zig-zags cancel out, the consistent direction adds up, so
        it races down the valley (and overshoots: watch it swing). <strong>Nesterov</strong> looks one step ahead and swings
        less. <strong>RMSprop</strong> divides each coordinate by a running RMS of its gradient, so steep and flat directions
        move at similar speeds. <strong>Adam</strong> combines both, with bias corrections for the first steps:</p>
        <p className="formula"><code>m ← β₁m + (1−β₁)g,  v ← β₂v + (1−β₂)g²,  p ← p − η · m̂ / (√v̂ + ε)</code></p>
        <p>Its first step is about η in every coordinate whatever the gradient's size, which is why Adam's learning rate means
        "distance per step" rather than "fraction of the gradient". On the <em>saddle</em>, start on the ridge (x ≈ 0): the
        gradient there is nearly zero along x, and methods without momentum crawl before falling into a minimum. Add
        <strong> gradient noise</strong> to see mini-batch training: the iterates never settle exactly, they jitter around the
        minimum. Every rule is written exactly as in torch.optim, and the Python tab checks that step for step.</p>
      </div>
    </div>
  );
}
