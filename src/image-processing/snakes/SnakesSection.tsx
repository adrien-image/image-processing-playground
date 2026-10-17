import { useEffect, useMemo, useState, type MouseEvent } from 'react';
import snakeCode from './snake.ts?raw';
import pythonCode from './snake.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { floatImage } from '../../deep-learning/shared/render';
import { circle, forceField, perimeter, resample, snakeMatrix, snakeStep, type Point } from './snake';
import './snakes.css';

const SIZE = 160;
const DISPLAY = 320;
const POINTS = 80;
const STEPS_PER_FRAME = 3;
const MAX_STEPS = 900;

/** A disc, a concave kidney shape, a square, and a blob with a weak, blurry edge. */
function demoImage(): HTMLCanvasElement {
  return makeCanvas(SIZE, SIZE, ctx => {
    ctx.fillStyle = '#26282e';
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = '#d6d6d6';
    ctx.beginPath(); ctx.arc(42, 42, 24, 0, Math.PI * 2); ctx.fill();
    // Kidney: a disc with a bite taken out, so the snake has to follow a concavity.
    ctx.beginPath(); ctx.arc(116, 46, 28, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#26282e';
    ctx.beginPath(); ctx.arc(140, 46, 15, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#bdbdbd';
    ctx.fillRect(22, 98, 46, 42);
    ctx.filter = 'blur(3px)';
    ctx.fillStyle = '#8a8a8a';
    ctx.beginPath(); ctx.ellipse(116, 118, 26, 18, 0.4, 0, Math.PI * 2); ctx.fill();
    ctx.filter = 'none';
  });
}

export function SnakesSection() {
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [start, setStart] = useState({ x: 42, y: 42, r: 36 });
  const [alpha, setAlpha] = useState(0.05);
  const [beta, setBeta] = useState(0.05);
  const [edgeWeight, setEdgeWeight] = useState(3);
  const [balloon, setBalloon] = useState(-0.4);
  const [sigma, setSigma] = useState(1.5);
  const [snake, setSnake] = useState<Point[]>(() => circle(42, 42, 36, POINTS));
  const [steps, setSteps] = useState(0);
  const [running, setRunning] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, SIZE);

  const field = useMemo(() => (image ? forceField(image.gray, image.width, image.height, sigma) : null), [image, sigma]);
  const inverse = useMemo(() => snakeMatrix(POINTS, alpha, beta, 1), [alpha, beta]);
  const edgeImage = useMemo(() => (field ? floatImage(field.edges.map(v => Math.sqrt(v)), field.width, field.height) : null), [field]);

  const restart = (s = start) => {
    setRunning(false);
    setSnake(circle(s.x, s.y, s.r, POINTS));
    setSteps(0);
  };

  const advance = (count: number) => {
    if (!field) return;
    let p = snake;
    for (let i = 0; i < count; i++) {
      p = snakeStep(p, inverse, field, { tau: 1, edgeWeight, balloon });
      if ((steps + i) % 10 === 9) p = resample(p);
    }
    setSnake(p);
    setSteps(s => s + count);
  };

  // Animation: a few steps per frame while running.
  const animating = running && steps < MAX_STEPS && field !== null;
  useEffect(() => {
    if (!animating) return;
    const id = requestAnimationFrame(() => advance(STEPS_PER_FRAME));
    return () => cancelAnimationFrame(id);
  });

  const place = (e: MouseEvent<HTMLDivElement>) => {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const s = { ...start, x: ((e.clientX - rect.left) / rect.width) * image.width, y: ((e.clientY - rect.top) / rect.height) * image.height };
    setStart(s);
    restart(s);
  };

  const w = image?.width ?? SIZE, h = image?.height ?? SIZE;
  const path = snake.map(p => `${p.x},${p.y}`).join(' ');
  const error = uploadError ?? loadError;
  const slider = (label: string, value: number, set: (v: number) => void, min: number, max: number, step: number, digits = 2) => (
    <label className="dl-slider">{label} <strong>{value.toFixed(digits)}</strong>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => set(Number(e.target.value))} />
    </label>
  );

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <button type="button" className="btn btn-sm btn-primary" onClick={() => setRunning(r => !r)} disabled={!field || steps >= MAX_STEPS}>
          {animating ? '⏸ Pause' : '▶ Run'}
        </button>
        <button type="button" className="btn btn-sm" onClick={() => advance(10)} disabled={animating}>+10 steps</button>
        <button type="button" className="btn btn-sm" onClick={() => restart()}>↺ Restart</button>
        <span className="demo-label">Start:</span>
        <button type="button" aria-pressed={balloon < 0} className={`btn btn-sm${balloon < 0 ? ' btn-primary' : ''}`}
          onClick={() => { const s = { ...start, r: 36 }; setBalloon(-0.4); setStart(s); restart(s); }}>Outside, shrink</button>
        <button type="button" aria-pressed={balloon > 0} className={`btn btn-sm${balloon > 0 ? ' btn-primary' : ''}`}
          onClick={() => { const s = { ...start, r: 5 }; setBalloon(0.4); setStart(s); restart(s); }}>Inside, grow</button>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={snakeCode} title="snakes/snake.ts" python={{ code: pythonCode, title: 'snakes/snake.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          {slider('Elasticity α', alpha, setAlpha, 0, 0.5, 0.01)}
          {slider('Rigidity β', beta, setBeta, 0, 0.5, 0.01)}
          {slider('Edge force w', edgeWeight, setEdgeWeight, 0, 8, 0.5, 1)}
          {slider('Balloon', balloon, setBalloon, -1, 1, 0.05)}
          {slider('Edge blur σ', sigma, setSigma, 0.5, 4, 0.25)}
          {slider('Start radius', start.r, r => { const s = { ...start, r }; setStart(s); restart(s); }, 3, 60, 1, 0)}
        </div>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Image · click an object to start there</h3>
          <div className="snake-picker" onClick={place} role="presentation">
            <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Image with the active contour; click to place the starting circle">
              <svg className="snake-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                <circle className="snake-start" cx={start.x} cy={start.y} r={start.r} />
                <polygon className="snake-curve" points={path} />
                {snake.map((p, i) => <circle key={i} className="snake-point" cx={p.x} cy={p.y} r={0.9} />)}
              </svg>
            </PixelCanvas>
          </div>
          <p className="caption">{steps} steps · perimeter {perimeter(snake).toFixed(0)} px · dashed: the starting circle</p>
        </div>
        <div className="demo-panel">
          <h3>Edge map |∇(G_σ ⋆ I)|</h3>
          <PixelCanvas image={edgeImage} size={DISPLAY} label="Edge strength, which the snake is pulled towards">
            <svg className="snake-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
              <polygon className="snake-curve" points={path} />
            </svg>
          </PixelCanvas>
          <p className="caption">The edge force points uphill on this map; it only reaches about σ from an edge</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>A rubber band that clings to edges</h4>
        <p>An active contour is a closed curve of points that moves to lower an energy with three parts:</p>
        <p className="formula"><code>E = α Σ |xᵢ₊₁ − xᵢ|² + β Σ |xᵢ₊₁ − 2xᵢ + xᵢ₋₁|² − w Σ |∇(G_σ ⋆ I)(xᵢ)|²</code></p>
        <p><strong>Elasticity α</strong> makes it want to be short, <strong>rigidity β</strong> makes it want to be smooth (no sharp
        bends), and the <strong>edge term</strong> rewards sitting on strong gradients. The first two are quadratic in the
        points, so their gradient is a matrix A times the points, and each step solves</p>
        <p className="formula"><code>(I + τA) · x_new = x + τ · F(x)</code></p>
        <p>with the inverse computed once. Treating the stiff internal forces implicitly keeps the step stable whatever α and β.</p>
        <p>The edge force only acts within a few σ of an edge: far away the image is flat and the snake just shrinks. A
        <strong> balloon</strong> force along the normal pushes it inward or outward until it meets an edge, which then holds it.
        Watch the trade-offs: high rigidity cannot follow the kidney's concavity, a too-strong balloon pushes through the weak
        blurry edge, and a larger σ reaches farther but rounds the corners of the square. Modern methods (level sets, deep
        segmentation) remove the need for a good starting curve, but snakes are still used for interactive tools.</p>
      </div>
    </div>
  );
}
