import { useState, useCallback, useRef, useEffect } from 'react';
import { loadImage, getImageData, generateCheckerboardDataURL } from '../utils/imageLoader';
import { rgbaToGrayscale, grayscaleToRGBA } from '../processors/utils';
import { computeHistogram, otsuThreshold, applyThreshold } from './otsu';
import { kmeans, labelsToImage, type KMeansResult } from './kmeans';
import { HistogramView } from './HistogramView';
import { CodeViewer } from '../components/CodeViewer';
import kmeansCode from './kmeans.ts?raw';

type DemoImage = 'bimodal' | 'tricluster' | 'gradient' | 'checkerboard';
type Method = 'otsu' | 'kmeans-2' | 'kmeans-3';

export function OtsuDemo() {
  const [showCode, setShowCode] = useState(false);
  const [pixels, setPixels] = useState<Float64Array | null>(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [histogram, setHistogram] = useState<Uint32Array | null>(null);
  const [otsuThresh, setOtsuThresh] = useState(0.5);
  const [userThresh, setUserThresh] = useState(0.5);
  const [demoImg, setDemoImg] = useState<DemoImage>('bimodal');
  const [method, setMethod] = useState<Method>('otsu');
  const [kmResult, setKmResult] = useState<KMeansResult | null>(null);

  const origRef = useRef<HTMLCanvasElement>(null);
  const segRef = useRef<HTMLCanvasElement>(null);
  const colourRef = useRef<HTMLCanvasElement>(null);

  const generateImage = useCallback(async (name: DemoImage) => {
    let src: string;
    const size = 64;

    if (name === 'checkerboard') {
      src = generateCheckerboardDataURL(size, 8);
    } else if (name === 'tricluster') {
      // Three vertical bands: dark, medium, light
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d')!;
      const bandW = size / 3;
      ctx.fillStyle = '#222'; ctx.fillRect(0, 0, bandW, size);
      ctx.fillStyle = '#888'; ctx.fillRect(bandW, 0, bandW, size);
      ctx.fillStyle = '#ddd'; ctx.fillRect(bandW * 2, 0, bandW, size);
      src = canvas.toDataURL();
    } else if (name === 'bimodal') {
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#ddd';
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = '#222';
      ctx.beginPath(); ctx.arc(size * 0.3, size * 0.3, size * 0.18, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(size * 0.7, size * 0.7, size * 0.15, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(size * 0.7, size * 0.25, size * 0.1, 0, Math.PI * 2); ctx.fill();
      src = canvas.toDataURL();
    } else {
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d')!;
      const gradient = ctx.createLinearGradient(0, 0, size, 0);
      gradient.addColorStop(0, '#000');
      gradient.addColorStop(1, '#fff');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);
      src = canvas.toDataURL();
    }

    const img = await loadImage(src);
    const { imageData, width: w, height: h } = getImageData(img, size);
    const gray = rgbaToGrayscale(imageData.data, w, h);
    const hist = computeHistogram(gray);
    const otsu = otsuThreshold(hist, w * h);

    setPixels(gray);
    setWidth(w);
    setHeight(h);
    setHistogram(hist);
    setOtsuThresh(otsu);
    setUserThresh(otsu);
    setDemoImg(name);
  }, []);

  useEffect(() => { generateImage('bimodal'); }, [generateImage]);

  // Draw canvases
  useEffect(() => {
    if (!pixels || !width || !height) return;

    const origCanvas = origRef.current;
    if (origCanvas) {
      origCanvas.width = width;
      origCanvas.height = height;
      const ctx = origCanvas.getContext('2d')!;
      const rgba = grayscaleToRGBA(pixels, width, height);
      ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
    }

    // Compute K-means synchronously
    const currentKmResult = method !== 'otsu' ? kmeans(pixels, method === 'kmeans-3' ? 3 : 2) : null;
    if (currentKmResult) setKmResult(currentKmResult);

    // Segmentation result canvas
    const segCanvas = segRef.current;
    if (segCanvas) {
      segCanvas.width = width;
      segCanvas.height = height;
      const ctx = segCanvas.getContext('2d')!;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, width, height);

      if (method === 'otsu') {
        const binary = applyThreshold(pixels, userThresh);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const v = binary[y * width + x] * 255;
            ctx.fillStyle = `rgb(${v},${v},${v})`;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      } else if (currentKmResult) {
        const segPixels = labelsToImage(pixels, currentKmResult.labels, currentKmResult.centroids);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const v = Math.round(segPixels[y * width + x]);
            ctx.fillStyle = `rgb(${v},${v},${v})`;
            ctx.fillRect(x, y, 1, 1);
          }
        }
      }
    }

    // Colour segmentation canvas (K-means only)
    const colCanvas = colourRef.current;
    if (colCanvas && currentKmResult && method !== 'otsu') {
      colCanvas.width = width;
      colCanvas.height = height;
      const ctx = colCanvas.getContext('2d')!;
      const colours: string[] = ['#6c8cff', '#f87171', '#34d399', '#fbbf24', '#c4b5fd'];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const label = currentKmResult.labels[y * width + x];
          ctx.fillStyle = colours[label % colours.length];
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }
  }, [pixels, width, height, userThresh, method]);

  const scale = 400 / Math.max(width, 1);
  const displayW = Math.round(width * scale);
  const displayH = Math.round(height * scale);

  return (
    <div className="demo-section">
      {/* Image picker + method selector */}
      <div className="demo-controls">
        <span className="demo-label">Demo image:</span>
        <button className={`btn btn-sm${demoImg === 'bimodal' ? ' btn-primary' : ''}`} onClick={() => generateImage('bimodal')}>Two‑tone</button>
        <button className={`btn btn-sm${demoImg === 'tricluster' ? ' btn-primary' : ''}`} onClick={() => generateImage('tricluster')}>Three‑band</button>
        <button className={`btn btn-sm${demoImg === 'gradient' ? ' btn-primary' : ''}`} onClick={() => generateImage('gradient')}>Gradient</button>
        <button className={`btn btn-sm${demoImg === 'checkerboard' ? ' btn-primary' : ''}`} onClick={() => generateImage('checkerboard')}>Checkerboard</button>
      </div>

      <div className="demo-controls">
        <span className="demo-label">Method:</span>
        <button className={`btn btn-sm${method === 'otsu' ? ' btn-primary' : ''}`} onClick={() => setMethod('otsu')}>Otsu Threshold</button>
        <button className={`btn btn-sm${method === 'kmeans-2' ? ' btn-primary' : ''}`} onClick={() => setMethod('kmeans-2')}>K‑means (k=2)</button>
        <button className={`btn btn-sm${method === 'kmeans-3' ? ' btn-primary' : ''}`} onClick={() => setMethod('kmeans-3')}>K‑means (k=3)</button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      {/* Images */}
      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <canvas ref={origRef} className="demo-canvas" style={{ width: displayW, height: displayH, imageRendering: 'pixelated' }} />
        </div>
        <div className="demo-panel">
          <h3>{method === 'otsu' ? 'Binary (threshold)' : `K‑means (k=${method === 'kmeans-3' ? 3 : 2})`}</h3>
          <canvas ref={segRef} className="demo-canvas" style={{ width: displayW, height: displayH, imageRendering: 'pixelated' }} />
        </div>
        {method !== 'otsu' && (
          <div className="demo-panel">
            <h3>Clusters (coloured)</h3>
            <canvas ref={colourRef} className="demo-canvas" style={{ width: displayW, height: displayH, imageRendering: 'pixelated' }} />
          </div>
        )}
      </div>

      {/* Threshold slider (Otsu only) */}
      {method === 'otsu' && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <span className="demo-slider-label">Threshold: <strong>{(userThresh * 255).toFixed(0)}</strong></span>
            <input type="range" min={0} max={1} step={0.005} value={userThresh}
              onChange={(e) => setUserThresh(Number(e.target.value))} className="demo-slider" />
            <span className="otsu-optimal" onClick={() => setUserThresh(otsuThresh)} style={{ cursor: 'pointer' }}>
              ⬤ Otsu: {(otsuThresh * 255).toFixed(0)}
            </span>
          </div>
        </div>
      )}

      {/* K-means centroid info */}
      {method !== 'otsu' && kmResult && (
        <div className="demo-slider-section">
          <div className="otsu-km-info">
            <span className="demo-label">Cluster centres:</span>
            {Array.from(kmResult.centroids).map((c, i) => (
              <span key={i} className="otsu-km-centroid" style={{ borderColor: ['#6c8cff', '#f87171', '#34d399'][i] || '#fbbf24' }}>
                <span className="otsu-km-swatch" style={{ background: ['#6c8cff', '#f87171', '#34d399'][i] || '#fbbf24' }} />
                C{i}: {c.toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Histogram (always visible) */}
      {histogram && (
        <HistogramView histogram={histogram} threshold={userThresh} otsuThreshold={otsuThresh} />
      )}

      {/* Explanation */}
      <div className="demo-explanation">
        {method === 'otsu' ? (
          <>
            <h4>How Otsu's Method Works</h4>
            <p>Otsu's method finds the optimal threshold by <strong>maximising between-class variance</strong>.
            It tries every possible threshold, splits the histogram into two parts, and picks the one with the best separation.</p>
            <div className="otsu-formula"><code>σ²<sub>between</sub>(t) = ω₁(t) · ω₂(t) · [μ₁(t) − μ₂(t)]²</code></div>
            <p>Works well for <strong>bimodal</strong> images (two distinct peaks). Struggles with images having more than two intensity populations.</p>
          </>
        ) : (
          <>
            <h4>How K‑means Clustering Works</h4>
            <p>K‑means partitions the pixels into <strong>k clusters</strong> by minimising the distance from each pixel to its cluster centre.
            The algorithm alternates between assigning pixels to the nearest centre and updating centres to the mean of their assigned pixels.</p>
            <div className="otsu-formula">
              <code>1. Assign: cᵢ = argmin‖xᵢ − μⱼ‖&emsp;2. Update: μⱼ = (1/|Cⱼ|) Σ xᵢ</code>
            </div>
            <p>Unlike Otsu, K‑means can separate <strong>more than two</strong> intensity levels. Try the <strong>Three‑band</strong> image with k=3 to see it in action.
            With k=2 on three bands, it merges the two closest bands — a common failure mode.</p>
          </>
        )}
        {demoImg === 'gradient' && (
          <p className="otsu-note">⚠ The gradient has a single smooth peak — segmentation methods struggle because there's no natural boundary.</p>
        )}
        {demoImg === 'checkerboard' && (
          <p className="otsu-note">⚠ The checkerboard has alternating black and white squares — thresholding separates them but the result is a checkerboard pattern, not a meaningful segmentation.</p>
        )}
        {demoImg === 'tricluster' && method === 'otsu' && (
          <p className="otsu-note">⚠ The three‑band image has <strong>three</strong> distinct intensity levels. Otsu can only find one threshold, so it merges two bands into one. K‑means with k=3 separates all three correctly.</p>
        )}
      </div>
      {showCode && <CodeViewer code={kmeansCode} title="kmeans.ts" onClose={() => setShowCode(false)} />}
    </div>
  );
}
