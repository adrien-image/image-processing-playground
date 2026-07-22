import { CodeViewer } from '../components/CodeViewer';
import histeqCode from './HistEqSection.tsx?raw';
import { useState, useEffect, useRef, useCallback } from 'react';
import { rgbaToGrayscale, grayscaleToRGBA } from '../processors/utils';
import { computeHistogram } from '../segmentation/otsu';
import { HistogramView } from '../segmentation/HistogramView';
import { ImageUploader } from '../components/ImageUploader';

function generateTestImage(): string {
  const c = document.createElement('canvas');
  c.width = 100; c.height = 100;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, 100, 100);
  const levels: [number, number, number, number, number][] = [
    [20, 20, 60, 60, 100], [12, 12, 30, 30, 60],
    [50, 50, 35, 35, 140], [70, 12, 20, 20, 200],
    [15, 55, 25, 25, 90], [55, 55, 30, 15, 170],
    [72, 60, 18, 25, 220], [10, 70, 15, 15, 50],
    [45, 12, 15, 18, 180], [60, 75, 30, 15, 130],
    [35, 72, 12, 12, 240], [80, 80, 15, 15, 75],
  ];
  for (const [x, y, w, h, level] of levels) {
    ctx.fillStyle = `rgb(${level},${level},${level})`;
    ctx.fillRect(x, y, w, h);
  }
  const grad = ctx.createRadialGradient(50, 60, 5, 50, 60, 20);
  grad.addColorStop(0, '#eee');
  grad.addColorStop(1, '#333');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(50, 60, 20, 0, Math.PI * 2); ctx.fill();
  return c.toDataURL();
}

export function HistEqSection() {
  const srcRef = useRef<HTMLImageElement>(null);
  const eqRef = useRef<HTMLCanvasElement>(null);
  const [histBefore, setHistBefore] = useState<Uint32Array | null>(null);
  const [histAfter, setHistAfter] = useState<Uint32Array | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [imageSrc, setImageSrc] = useState<string>(generateTestImage());

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

    // Histogram before (adaptive bins — for display)
    setHistBefore(computeHistogram(gray));

    // Compute a proper 256-bin histogram for the CDF
    const hist = new Float64Array(256);
    for (let i = 0; i < w * h; i++) hist[Math.round(gray[i])]++;

    const cdf = new Float64Array(256);
    cdf[0] = hist[0];
    for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];

    // Find minimum non-zero CDF value for proper full-range stretching
    let cdfMin = 0;
    for (let i = 0; i < 256; i++) {
      if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
    }

    const total = w * h;
    const denom = total - cdfMin || 1;

    // Equalise: map each pixel via (cdf[val] - cdfMin) / (total - cdfMin) * 255
    const eq = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const idx = Math.round(gray[i]);
      eq[i] = ((cdf[idx] - cdfMin) / denom) * 255;
    }

    // Histogram after
    setHistAfter(computeHistogram(eq));

    // Draw equalised image
    const eCanvas = eqRef.current;
    if (eCanvas) {
      eCanvas.width = w; eCanvas.height = h;
      const eCtx = eCanvas.getContext('2d')!;
      const rgba = grayscaleToRGBA(eq, w, h);
      eCtx.putImageData(new ImageData(rgba, w, h), 0, 0);
    }
  }, []);

  // Run process whenever the image loads
  const handleImageLoad = useCallback(() => { process(); }, [process]);

  useEffect(() => {
    const img = srcRef.current;
    if (!img) return;
    img.onload = handleImageLoad;
    if (img.complete && img.naturalWidth) process();
    return () => { img.onload = null; };
  }, [imageSrc, handleImageLoad, process]);

  const handleUpload = useCallback((img: HTMLImageElement) => {
    setImageSrc(img.src);
  }, []);

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Image:</span>
        <ImageUploader onImage={handleUpload} label="📁 Upload photo" />
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>
      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original (low contrast)</h3>
          <img ref={srcRef} src={imageSrc} className="color-img" alt="original" style={{ width: 200, height: 200 }} />
        </div>
        <div className="demo-panel">
          <h3>Equalised</h3>
          <canvas ref={eqRef} className="color-img" style={{ width: 200, height: 200 }} />
        </div>
      </div>

      <div className="demo-slider-section" style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--accent)', marginBottom: '0.4rem' }}>Original histogram</div>
          {histBefore && <HistogramView histogram={histBefore} threshold={-1} otsuThreshold={-1} width={280} height={140} />}
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--accent)', marginBottom: '0.4rem' }}>Equalised histogram</div>
          {histAfter && <HistogramView histogram={histAfter} threshold={-1} otsuThreshold={-1} width={280} height={140} />}
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Histogram Equalisation</h4>
        <p>Histogram equalisation spreads out the most frequent intensity values to improve contrast.
        It computes the <strong>Cumulative Distribution Function (CDF)</strong> of the pixel intensities,
        then maps each pixel to its percentile rank × 255.</p>
        <p>The CDF is normalised so the lowest occupied intensity maps to 0 and the highest to 255,
        stretching the output to the full dynamic range.</p>
        <p>The result: dark pixels become darker, light pixels become lighter, and the mid-range is stretched — revealing details that were previously hard to see.</p>
        <p>Compare the two histograms: the original is narrow (pixels concentrated in a small range), while the equalised histogram is spread across the full 0–255 range.</p>
      </div>
    </div>
      {showCode && <CodeViewer code={histeqCode} title="HistEqSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
