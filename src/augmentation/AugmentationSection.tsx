import { CodeViewer } from '../components/CodeViewer';
import augmentCode from './augment.ts?raw';
import { useState, useCallback, useRef, useEffect } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { generateAugmentConfigs, type AugmentConfig } from './augment';

function generateDemoImage(): string {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 128;
  const ctx = c.getContext('2d')!;
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
  return c.toDataURL();
}

export function AugmentationSection() {
  const [imageSrc, setImageSrc] = useState(generateDemoImage());
  const [showCode, setShowCode] = useState(false);
  const [results, setResults] = useState<{ label: string; src: string }[]>([]);
  const [opts, setOpts] = useState({
    rotate: true, flip: true, crop: true, brightness: true,
    contrast: false, gaussianNoise: false, saltPepper: false, shear: false,
  });
  const imgRef = useRef<HTMLImageElement>(null);

  const toggle = (key: keyof typeof opts) => setOpts(prev => ({ ...prev, [key]: !prev[key] }));

  const generate = useCallback(async () => {
    const img = imgRef.current;
    if (!img || !img.complete) return;
    const configs = generateAugmentConfigs(opts);
    // Apply sequentially: each transform receives the output of the previous one
    const outputs: { label: string; src: string }[] = [];
    let current = img;
    const appliedLabels: string[] = [];
    for (const cfg of configs) {
      const src = cfg.apply(current);
      appliedLabels.push(cfg.label);
      outputs.push({ label: appliedLabels.join(' → '), src });
      // Create a new Image for the next iteration
      current = new Image();
      current.src = src;
      await new Promise<void>((resolve) => { current.onload = () => resolve(); });
    }
    setResults(outputs);
  }, [opts]);

  const handleUpload = useCallback((img: HTMLImageElement) => {
    setImageSrc(img.src);
    setResults([]);
  }, []);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const onLoad = () => generate();
    img.onload = onLoad;
    if (img.complete && img.naturalWidth) generate();
  }, [imageSrc, generate]);

  const displayW = 160, displayH = 160;

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Pipeline (applied in order):</span>
        {([
          ['rotate', '↻ Rotate'], ['flip', '↔ Flip'], ['crop', '⊞ Crop'],
          ['brightness', '☀ Brightness'], ['contrast', '◑ Contrast'],
          ['gaussianNoise', '░ Gauss Noise'], ['saltPepper', '▚ S&P Noise'],
          ['shear', '⫽ Shear'],
        ] as [keyof typeof opts, string][]).map(([k, label]) => (
          <button key={k}
            className={`btn btn-sm${opts[k] ? ' btn-primary' : ''}`}
            style={opts[k] ? {} : { opacity: 0.45 }}
            onClick={() => toggle(k)}>
            {opts[k] ? '✓ ' : '✗ '}{label}
          </button>
        ))}
      </div>

      <div className="demo-controls">
        <span className="demo-label">Image:</span>
        <ImageUploader onImage={handleUpload} label="📁 Upload" />
        <button className="btn btn-primary btn-sm" onClick={generate} style={{ marginLeft: '0.5rem' }}>
          🎲 Regenerate
        </button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      <div className="demo-images" style={{ justifyContent: 'center' }}>
        <div className="demo-panel">
          <h3>Original</h3>
          <img ref={imgRef} src={imageSrc} alt="original"
            style={{ width: displayW, height: displayH, objectFit: 'contain', borderRadius: 'var(--radius)', background: '#000' }} />
        </div>

        {results.map((r, i) => (
          <div key={i} className="demo-panel">
            <h3>{r.label}</h3>
            <img src={r.src} alt={r.label}
              style={{ width: displayW, height: displayH, objectFit: 'contain', borderRadius: 'var(--radius)', background: '#000' }} />
          </div>
        ))}

        {results.length === 0 && (
          <div className="demo-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: displayH }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              Enable augmentations above and click 🎲 Regenerate
            </span>
          </div>
        )}
      </div>

      <div className="demo-explanation">
        <h4>Data Augmentation</h4>
        <p>Data augmentation artificially expands a training dataset by applying random transformations to existing images.
        This helps ML models generalise better by seeing variations of the same sample — different angles, lighting, noise levels.</p>
        <p><strong>Rotate</strong> — random rotation within a small angle range. <strong>Flip</strong> — horizontal or vertical mirror.
        <strong>Crop</strong> — random region extracted and resized. <strong>Brightness/Contrast</strong> — pixel value scaling.
        <strong>Gaussian noise</strong> — simulates sensor noise. <strong>S&P noise</strong> — dead/stuck pixels.
        <strong>Shear</strong> — affine skew along the horizontal axis.</p>
        <p>Transformations are applied <strong>accumulatively</strong> — each one feeds into the next, so you can see the compound effect (e.g. Rotate → Flip → Noise).
        Enable the augmentations you want and click <strong>🎲 Regenerate</strong> to see new random variants.</p>
      </div>
    </div>
      {showCode && <CodeViewer code={augmentCode} title="augment.ts" onClose={() => setShowCode(false)} />}
    </>
  );
}
