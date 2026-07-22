import { useState, useRef, useEffect, useCallback } from 'react';
import { rgbaToGrayscale, grayscaleToRGBA } from '../processors/utils';
import { CodeViewer } from '../components/CodeViewer';
import denoiseCode from './DenoisingSection.tsx?raw';

function generateImage(): string {
  const c = document.createElement('canvas');
  c.width = 128; c.height = 128;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, 128, 128);
  ctx.fillStyle = '#ddd';
  ctx.fillRect(20, 20, 50, 50);
  ctx.fillRect(80, 30, 30, 40);
  ctx.beginPath(); ctx.arc(80, 90, 25, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#aaa';
  ctx.fillRect(15, 80, 25, 35);
  return c.toDataURL();
}

/** Median filter: each pixel = median of its neighborhood */
function medianFilter(pixels: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  const size = (radius * 2 + 1) ** 2;
  const window = new Float64Array(size);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let wi = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const px = x + dx, py = y + dy;
          window[wi++] = (px >= 0 && px < w && py >= 0 && py < h)
            ? pixels[py * w + px] : 0;
        }
      }
      window.sort();
      out[y * w + x] = window[Math.floor(size / 2)];
    }
  }
  return out;
}

/** Add salt-and-pepper noise */
function addSaltPepper(pixels: Float64Array, density: number): Float64Array {
  const out = new Float64Array(pixels);
  for (let i = 0; i < out.length; i++) {
    const r = Math.random();
    if (r < density / 2) out[i] = 0;
    else if (r < density) out[i] = 255;
  }
  return out;
}

/** Gaussian blur (3×3) */
function gaussianBlur(pixels: Float64Array, w: number, h: number): Float64Array {
  const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  const out = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0, ki = 0;
      for (let ky = -1; ky <= 1; ky++)
        for (let kx = -1; kx <= 1; kx++)
          sum += pixels[(y + ky) * w + (x + kx)] * kernel[ki++];
      out[y * w + x] = sum / 16;
    }
  }
  return out;
}

type FilterType = 'median' | 'gaussian';

export function DenoisingSection() {
  const [filter, setFilter] = useState<FilterType>('median');
  const [noiseDensity, setNoiseDensity] = useState(0.08);
  const [radius, setRadius] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const srcRef = useRef<HTMLImageElement>(null);
  const noisyRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef<HTMLCanvasElement>(null);

  const process = useCallback(() => {
    const img = srcRef.current;
    if (!img || !img.complete || !img.naturalWidth) return;
    const w = img.naturalWidth, h = img.naturalHeight;

    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, w, h);
    const gray = rgbaToGrayscale(imageData.data, w, h);

    // Add noise
    const noisy = addSaltPepper(gray, noiseDensity);
    const nCanvas = noisyRef.current;
    if (nCanvas) {
      nCanvas.width = w; nCanvas.height = h;
      const nCtx = nCanvas.getContext('2d')!;
      const rgba = grayscaleToRGBA(noisy, w, h);
      nCtx.putImageData(new ImageData(rgba, w, h), 0, 0);
    }

    // Denoise
    const denoised = filter === 'median'
      ? medianFilter(noisy, w, h, radius)
      : gaussianBlur(noisy, w, h);

    const rCanvas = resultRef.current;
    if (rCanvas) {
      rCanvas.width = w; rCanvas.height = h;
      const rCtx = rCanvas.getContext('2d')!;
      const rgba = grayscaleToRGBA(denoised, w, h);
      rCtx.putImageData(new ImageData(rgba, w, h), 0, 0);
    }
  }, [filter, noiseDensity, radius]);

  useEffect(() => {
    const img = srcRef.current;
    if (!img) return;
    const onLoad = () => process();
    img.onload = onLoad;
    img.src = generateImage();
    if (img.complete && img.naturalWidth) process();
    return () => { img.onload = null; };
  }, [process]);

  const disp = 256;

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Filter:</span>
        <button className={`btn btn-sm${filter === 'median' ? ' btn-primary' : ''}`} onClick={() => setFilter('median')}>Median</button>
        <button className={`btn btn-sm${filter === 'gaussian' ? ' btn-primary' : ''}`} onClick={() => setFilter('gaussian')}>Gaussian</button>
        <span className="demo-label" style={{ marginLeft: '1rem' }}>Noise:</span>
        <input type="range" min={0.01} max={0.3} step={0.01} value={noiseDensity}
          onChange={e => setNoiseDensity(Number(e.target.value))} className="demo-slider" style={{ width: 120 }} />
        <span className="demo-slider-label">{(noiseDensity * 100).toFixed(0)}%</span>
        {filter === 'median' && (
          <>
            <span className="demo-label" style={{ marginLeft: '1rem' }}>Radius: {radius}</span>
            <input type="range" min={1} max={3} value={radius}
              onChange={e => setRadius(Number(e.target.value))} className="demo-slider" style={{ width: 80 }} />
          </>
        )}
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <img ref={srcRef} className="color-img" alt="original" style={{ width: disp, height: disp }} />
        </div>
        <div className="demo-panel">
          <h3>Noisy ({filter === 'median' ? 'Salt & Pepper' : 'Gaussian noisy'})</h3>
          <canvas ref={noisyRef} className="color-img" style={{ width: disp, height: disp }} />
        </div>
        <div className="demo-panel">
          <h3>{filter === 'median' ? `Median ${(radius * 2 + 1)}×${(radius * 2 + 1)}` : 'Gaussian 3×3'}</h3>
          <canvas ref={resultRef} className="color-img" style={{ width: disp, height: disp }} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>{filter === 'median' ? 'Median Filter' : 'Gaussian Blur'}</h4>
        {filter === 'median' ? (
          <>
            <p>The <strong>median filter</strong> replaces each pixel with the median value of its neighborhood.
            Unlike Gaussian blur which averages (blurring edges), the median preserves sharp edges while removing isolated noise pixels.</p>
            <p>It's the go-to filter for <strong>salt-and-pepper noise</strong> — those random black/white specks.
            Increase the noise density slider and compare median vs Gaussian: the median keeps edges crisp while Gaussian smears them.</p>
            <p>Increase the radius to see how a larger window removes more noise but can erode small details.</p>
          </>
        ) : (
          <>
            <p><strong>Gaussian blur</strong> replaces each pixel with a weighted average of its neighborhood.
            It's great for Gaussian noise (sensor noise) but smears edges when dealing with salt-and-pepper noise.</p>
            <p>Compare the two: on salt-and-pepper noise, median preserves sharp edges while Gaussian creates blurry halos around noise specks.</p>
          </>
        )}
      </div>
    </div>
    {showCode && <CodeViewer code={denoiseCode} title="DenoisingSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
