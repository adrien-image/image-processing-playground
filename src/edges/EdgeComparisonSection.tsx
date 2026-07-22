import { CodeViewer } from '../components/CodeViewer';
import edgesCode from './EdgeComparisonSection.tsx?raw';
import { useState, useRef, useEffect, useCallback } from 'react';
import { getImageData } from '../utils/imageLoader';
import { rgbaToGrayscale, grayscaleToRGBA, convolve, normaliseMagnitude } from '../processors/utils';
import { SOBEL_X, SOBEL_Y } from '../processors/sobel/kernels';
import { ImageUploader } from '../components/ImageUploader';

const LAPLACIAN_3x3: number[][] = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]];
const PREWITT_X: number[][] = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
const PREWITT_Y: number[][] = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]];
const GAUSSIAN_5x5: number[][] = [
  [1,  4,  7,  4, 1],
  [4, 16, 26, 16, 4],
  [7, 26, 41, 26, 7],
  [4, 16, 26, 16, 4],
  [1,  4,  7,  4, 1],
];

function generateTestImage(): string {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 128;
  const ctx = c.getContext('2d')!;
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
  return c.toDataURL();
}

type EdgeMethod = 'sobel-gx' | 'sobel-gy' | 'sobel-mag' | 'laplacian' | 'prewitt' | 'canny';

/** Compute gradient magnitude image from Gx and Gy */
function gradientMagnitude(gx: Float64Array, gy: Float64Array): Float64Array {
  const n = gx.length;
  const mag = new Float64Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
  return mag;
}


/**
 * Canny edge detector.
 * 1. Gaussian blur → 2. Sobel gradient → 3. Non-max suppression →
 * 4. Double threshold → 5. Hysteresis edge tracking
 */
function canny(pixels: Float64Array, w: number, h: number, low: number, high: number): Float64Array {
  // 1. Gaussian blur
  let blurred = convolve(pixels, w, h, GAUSSIAN_5x5);
  // Normalise (Gaussian 5×5 sum is 273)
  const gaussSum = 273;
  for (let i = 0; i < blurred.length; i++) blurred[i] /= gaussSum;

  // 2. Sobel gradients
  const gx = convolve(blurred, w, h, SOBEL_X);
  const gy = convolve(blurred, w, h, SOBEL_Y);

  // Gradient magnitude + direction
  const mag = gradientMagnitude(gx, gy);
  const dir = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    dir[i] = Math.atan2(gy[i], gx[i]); // [-π, π]
  }

  // 3. Non-maximum suppression
  const nms = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const angle = dir[idx];
      // Quantize direction to 4 orientations: 0°, 45°, 90°, 135°
      let a = angle * (180 / Math.PI);
      if (a < 0) a += 180; // [0, 180)
      const m = mag[idx];

      let n1 = 0, n2 = 0;
      if (a < 22.5 || a >= 157.5) {          // horizontal edge → compare E/W
        n1 = mag[y * w + (x - 1)];
        n2 = mag[y * w + (x + 1)];
      } else if (a >= 22.5 && a < 67.5) {     // 45° diagonal → compare NE/SW
        n1 = mag[(y - 1) * w + (x + 1)];
        n2 = mag[(y + 1) * w + (x - 1)];
      } else if (a >= 67.5 && a < 112.5) {    // vertical edge → compare N/S
        n1 = mag[(y - 1) * w + x];
        n2 = mag[(y + 1) * w + x];
      } else {                                 // 135° diagonal → compare NW/SE
        n1 = mag[(y - 1) * w + (x - 1)];
        n2 = mag[(y + 1) * w + (x + 1)];
      }
      nms[idx] = (m >= n1 && m >= n2) ? m : 0;
    }
  }

  // 4. Double threshold
  // Scale low/high to the actual gradient range
  let maxMag = 0;
  for (let i = 0; i < w * h; i++) if (nms[i] > maxMag) maxMag = nms[i];
  const tLow = maxMag * (low / 100);
  const tHigh = maxMag * (high / 100);

  const strong = 255, weak = 75;
  const edges = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= tHigh) edges[i] = strong;
    else if (nms[i] >= tLow) edges[i] = weak;
  }

  // 5. Hysteresis: weak pixels adjacent to strong become strong
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      if (edges[idx] !== weak) continue;
      // Check 8-neighborhood for any strong pixel
      let hasStrong = false;
      for (let dy = -1; dy <= 1 && !hasStrong; dy++)
        for (let dx = -1; dx <= 1 && !hasStrong; dx++)
          if (edges[(y + dy) * w + (x + dx)] === strong) hasStrong = true;
      edges[idx] = hasStrong ? strong : 0;
    }
  }

  return edges;
}

function detectEdge(pixels: Float64Array, w: number, h: number, method: EdgeMethod, cannyLow = 20, cannyHigh = 50): Float64Array {
  if (method === 'sobel-gx') return normaliseMagnitude(convolve(pixels, w, h, SOBEL_X));
  if (method === 'sobel-gy') return normaliseMagnitude(convolve(pixels, w, h, SOBEL_Y));
  if (method === 'laplacian') return normaliseMagnitude(convolve(pixels, w, h, LAPLACIAN_3x3));
  if (method === 'prewitt') {
    const gx = convolve(pixels, w, h, PREWITT_X);
    const gy = convolve(pixels, w, h, PREWITT_Y);
    return normaliseMagnitude(gradientMagnitude(gx, gy));
  }
  if (method === 'canny') {
    return canny(pixels, w, h, cannyLow, cannyHigh);
  }
  // sobel-mag
  const gx = convolve(pixels, w, h, SOBEL_X);
  const gy = convolve(pixels, w, h, SOBEL_Y);
  return normaliseMagnitude(gradientMagnitude(gx, gy));
}

const labels: Record<EdgeMethod, string> = {
  'sobel-gx': 'Sobel Gx (vertical)',
  'sobel-gy': 'Sobel Gy (horizontal)',
  'sobel-mag': 'Sobel Magnitude',
  'laplacian': 'Laplacian',
  'prewitt': 'Prewitt',
  'canny': 'Canny',
};

const descriptions: Record<EdgeMethod, string> = {
  'sobel-gx': 'Detects vertical edges (bright↔dark transitions left/right)',
  'sobel-gy': 'Detects horizontal edges (bright↔dark transitions top/bottom)',
  'sobel-mag': 'Combines Gx and Gy — detects all orientations. The most common Sobel output.',
  'laplacian': 'Second-derivative operator. Responds to intensity changes on both sides of an edge. No directional preference.',
  'prewitt': 'Simpler than Sobel (all kernel values are ±1 instead of weighted). More noise-sensitive.',
  'canny': 'Multi-stage: Gaussian blur → Sobel gradient → Non-max suppression → Double threshold → Hysteresis. Produces thin, connected edges.',
};

export function EdgeComparisonSection() {
  const [method, setMethod] = useState<EdgeMethod>('sobel-mag');
  const [cannyLow, setCannyLow] = useState(20);
  const [showCode, setShowCode] = useState(false);
  const [cannyHigh, setCannyHigh] = useState(50);
  const srcRef = useRef<HTMLImageElement>(null);
  const dstRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (srcRef.current) srcRef.current.src = generateTestImage();
  }, []);

  const process = useCallback(async () => {
    const img = srcRef.current;
    const canvas = dstRef.current;
    if (!img || !canvas) return;
    if (!img.complete || !img.naturalWidth) {
      await new Promise<void>(r => {
        img.onload = () => r();
        // If it completed between our check and assigning onload, resolve now
        if (img.complete) r();
      });
    }
    const { imageData, width, height } = getImageData(img, 128);
    const gray = rgbaToGrayscale(imageData.data, width, height);
    const edges = detectEdge(gray, width, height, method, cannyLow, cannyHigh);
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const rgba = grayscaleToRGBA(edges, width, height);
    ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  }, [method, cannyLow, cannyHigh]);

  useEffect(() => { process(); }, [process]);

  const scale = 300 / 128;
  const dispW = Math.round(128 * scale);
  const dispH = Math.round(128 * scale);

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Edge detector:</span>
        {(Object.keys(labels) as EdgeMethod[]).map(m => (
          <button key={m} className={`btn btn-sm${method === m ? ' btn-primary' : ''}`} onClick={() => setMethod(m)}>
            {m === 'sobel-gx' ? 'Sobel Gx' : m === 'sobel-gy' ? 'Sobel Gy' : m === 'sobel-mag' ? 'Sobel Mag' : m === 'laplacian' ? 'Laplacian' : m === 'prewitt' ? 'Prewitt' : 'Canny'}
          </button>
        ))}
        <ImageUploader onImage={(img) => { if (srcRef.current) srcRef.current.src = img.src; }} label="📁 Upload" />
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      {method === 'canny' && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <span className="demo-slider-label">Low threshold: <strong>{cannyLow}%</strong></span>
            <input type="range" min={5} max={95} value={cannyLow}
              onChange={e => { const v = Number(e.target.value); setCannyLow(v); if (v >= cannyHigh) setCannyHigh(Math.min(95, v + 10)); }}
              className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <span className="demo-slider-label">High threshold: <strong>{cannyHigh}%</strong></span>
            <input type="range" min={10} max={95} value={cannyHigh}
              onChange={e => { const v = Number(e.target.value); setCannyHigh(v); if (v <= cannyLow) setCannyLow(Math.max(5, v - 10)); }}
              className="demo-slider" />
          </div>
        </div>
      )}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <img ref={srcRef} className="color-img" alt="original" />
        </div>
        <div className="demo-panel">
          <h3>{labels[method]}</h3>
          <canvas ref={dstRef} className="color-img" style={{ width: dispW, height: dispH, imageRendering: 'pixelated' }} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>{labels[method]}</h4>
        <p>{descriptions[method]}</p>
        {method === 'canny' && (
          <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            Weak edges (below low threshold) are discarded. Pixels between low and high thresholds are kept only if connected to a strong edge.
          </p>
        )}
        <p>Try each detector on the same image to compare how different operators highlight edges.</p>
      </div>
    </div>
      {showCode && <CodeViewer code={edgesCode} title="EdgeComparisonSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
