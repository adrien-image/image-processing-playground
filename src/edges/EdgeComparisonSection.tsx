import { useMemo, useState } from 'react';
import edgesCode from './edgeDetectors.ts?raw';
import { ShowCodeButton } from '../components/CodeViewer';
import { PixelCanvas } from '../components/PixelCanvas';
import { ImageUploader } from '../components/ImageUploader';
import { useImageData, type ImageSource } from '../hooks/useImageData';
import { makeCanvas } from '../lib/demoImages';
import { grayToImageData } from '../lib/image';
import { detectEdges, type EdgeMethod } from './edgeDetectors';

function testImage(): HTMLCanvasElement {
  return makeCanvas(128, 128, ctx => {
    ctx.fillStyle = '#333';
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = '#ddd';
    ctx.fillRect(20, 20, 40, 40);
    ctx.beginPath(); ctx.arc(85, 85, 30, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#bbb';
    ctx.fillRect(20, 80, 30, 30);
    ctx.fillRect(80, 15, 30, 18);
    ctx.fillStyle = '#888';
    ctx.beginPath(); ctx.arc(40, 105, 8, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(105, 40, 8, 0, Math.PI * 2); ctx.fill();
  });
}

const METHODS: { key: EdgeMethod; button: string; label: string; description: string }[] = [
  { key: 'sobel-gx', button: 'Sobel Gx', label: 'Sobel Gx (vertical)', description: 'Detects vertical edges (bright↔dark transitions left/right).' },
  { key: 'sobel-gy', button: 'Sobel Gy', label: 'Sobel Gy (horizontal)', description: 'Detects horizontal edges (bright↔dark transitions top/bottom).' },
  { key: 'sobel-mag', button: 'Sobel Mag', label: 'Sobel Magnitude', description: 'Combines Gx and Gy as √(Gx² + Gy²) — detects all orientations. The most common Sobel output.' },
  { key: 'laplacian', button: 'Laplacian', label: 'Laplacian', description: 'Second-derivative operator. Responds on both sides of an edge (a double line), with no directional preference, and is sensitive to noise.' },
  { key: 'prewitt', button: 'Prewitt', label: 'Prewitt', description: 'Like Sobel but with uniform weights (all ±1) instead of [1, 2, 1] smoothing — slightly more noise-sensitive.' },
  { key: 'canny', button: 'Canny', label: 'Canny', description: 'Multi-stage: Gaussian blur → Sobel gradient → non-max suppression along the gradient → double threshold → hysteresis. Produces thin, connected edges.' },
];

const DISPLAY = 300;

export function EdgeComparisonSection() {
  const [method, setMethod] = useState<EdgeMethod>('sobel-mag');
  const [cannyLow, setCannyLow] = useState(20);
  const [cannyHigh, setCannyHigh] = useState(50);
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(testImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 256);

  const edges = useMemo(() => {
    if (!image) return null;
    const out = detectEdges(image.gray, image.width, image.height, method, cannyLow, cannyHigh);
    return grayToImageData(out, image.width, image.height);
  }, [image, method, cannyLow, cannyHigh]);

  const info = METHODS.find(m => m.key === method)!;
  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Edge detector:</span>
        {METHODS.map(m => (
          <button key={m.key} type="button" aria-pressed={method === m.key}
            className={`btn btn-sm${method === m.key ? ' btn-primary' : ''}`} onClick={() => setMethod(m.key)}>
            {m.button}
          </button>
        ))}
        <ImageUploader label="📁 Upload" onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(testImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={edgesCode} title="edges/edgeDetectors.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {method === 'canny' && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="canny-low">Low threshold: <strong>{cannyLow}%</strong></label>
            <input id="canny-low" type="range" min={5} max={90} value={cannyLow} className="demo-slider"
              onChange={e => { const v = Number(e.target.value); setCannyLow(v); if (v >= cannyHigh) setCannyHigh(Math.min(95, v + 5)); }} />
          </div>
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="canny-high">High threshold: <strong>{cannyHigh}%</strong></label>
            <input id="canny-high" type="range" min={10} max={95} value={cannyHigh} className="demo-slider"
              onChange={e => { const v = Number(e.target.value); setCannyHigh(v); if (v <= cannyLow) setCannyLow(Math.max(5, v - 5)); }} />
          </div>
        </div>
      )}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Original image" />
        </div>
        <div className="demo-panel">
          <h3>{info.label}</h3>
          <PixelCanvas image={edges} size={DISPLAY} label={`${info.label} edges`} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>{info.label}</h4>
        <p>{info.description}</p>
        {method === 'canny' && (
          <p className="muted">
            Pixels above the high threshold are edges. Pixels between the two thresholds are kept only if they connect —
            directly or through other such pixels — to a strong edge; everything below the low threshold is discarded.
          </p>
        )}
        <p>Try each detector on the same image to compare how different operators highlight edges.</p>
      </div>
    </div>
  );
}
