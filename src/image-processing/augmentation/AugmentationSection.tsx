import { useMemo, useState } from 'react';
import augmentCode from './augment.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { canvasToImageData, imageDataToCanvas } from '../../lib/canvas';
import { seededRandom } from '../../lib/random';
import { sampleAugmentations, type AugmentationType } from './augment';
import './augmentation.css';

function demoImage(): HTMLCanvasElement {
  return makeCanvas(128, 128, ctx => {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = '#e94560';
    ctx.fillRect(10, 10, 50, 50);
    ctx.fillStyle = '#0f3460';
    ctx.beginPath(); ctx.arc(100, 100, 25, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#16c79a';
    ctx.fillRect(60, 60, 40, 20);
    ctx.fillStyle = '#f5c518';
    ctx.beginPath(); ctx.arc(40, 90, 15, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px monospace';
    ctx.fillText('DATA', 65, 40);
  });
}

const BUTTONS: [AugmentationType, string][] = [
  ['rotate', '↻ Rotate'], ['flip', '↔ Flip'], ['crop', '⊞ Crop'],
  ['brightness', '☀ Brightness'], ['contrast', '◑ Contrast'],
  ['gaussianNoise', '░ Gauss Noise'], ['saltPepper', '▚ S&P Noise'],
  ['shear', '⫽ Shear'],
];

const DISPLAY = 160;

export function AugmentationSection() {
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [seed, setSeed] = useState(1);
  const [enabled, setEnabled] = useState<Record<AugmentationType, boolean>>({
    rotate: true, flip: true, crop: true, brightness: true,
    contrast: false, gaussianNoise: false, saltPepper: false, shear: false,
  });
  const { image, error: loadError } = useImageData(source, 256);

  // Results are a pure function of (image, enabled steps, seed): toggling a step or
  // pressing Regenerate recomputes synchronously — no async image loading.
  const results = useMemo(() => {
    if (!image) return [];
    const steps = sampleAugmentations(enabled, seededRandom(seed));
    let current = imageDataToCanvas(image.imageData);
    const labels: string[] = [];
    return steps.map(step => {
      current = step.apply(current);
      labels.push(step.label);
      return { label: labels.join(' → '), image: canvasToImageData(current) };
    });
  }, [image, enabled, seed]);

  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Pipeline (applied in order):</span>
        {BUTTONS.map(([k, label]) => (
          <button key={k} type="button" aria-pressed={enabled[k]}
            className={`btn btn-sm${enabled[k] ? ' btn-primary' : ' btn-off'}`}
            onClick={() => setEnabled(prev => ({ ...prev, [k]: !prev[k] }))}>
            {enabled[k] ? '✓ ' : '✗ '}{label}
          </button>
        ))}
      </div>

      <div className="demo-controls">
        <span className="demo-label">Image:</span>
        <ImageUploader onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} label="📁 Upload" />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>Demo image</button>}
        <button type="button" className="btn btn-primary btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 Regenerate</button>
        <span className="ml-auto">
          <ShowCodeButton code={augmentCode} title="augmentation/augment.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Original image" />
        </div>

        {results.map((r, i) => (
          <div key={i} className="demo-panel augment-panel">
            <h3>{r.label}</h3>
            <PixelCanvas image={r.image} size={DISPLAY} pixelated={false} label={`After ${r.label}`} />
          </div>
        ))}

        {results.length === 0 && (
          <div className="demo-panel augment-empty">
            <span className="muted">Enable at least one augmentation above.</span>
          </div>
        )}
      </div>

      <div className="demo-explanation">
        <h4>Data Augmentation</h4>
        <p>Data augmentation artificially expands a training dataset by applying random transformations to existing images.
        This helps ML models generalise better by seeing variations of the same sample — different angles, lighting, noise levels.</p>
        <p><strong>Rotate</strong> — random angle within ±30°, keeping the image size (corners become black). <strong>Flip</strong> — horizontal or vertical mirror.
        <strong> Crop</strong> — random 70–90% window resized back up. <strong>Brightness/Contrast</strong> — pixel value scaling.
        <strong> Gaussian noise</strong> — simulates sensor noise. <strong>S&amp;P noise</strong> — dead/stuck pixels.
        <strong> Shear</strong> — affine skew along the horizontal axis.</p>
        <p>Transformations are applied <strong>cumulatively</strong> — each one feeds into the next, so you can see the compound effect.
        Each panel's title shows the exact parameters drawn. Click <strong>🎲 Regenerate</strong> for new random values.</p>
      </div>
    </div>
  );
}
