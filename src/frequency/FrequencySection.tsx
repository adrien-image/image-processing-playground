import { CodeViewer } from '../components/CodeViewer';
import fftCode from './fft.ts?raw';
import { useState, useEffect, useRef, useCallback } from 'react';
import { rgbaToGrayscale, grayscaleToRGBA } from '../processors/utils';
import { fft2d, ifft2d, pixelsToComplex, magnitudeSpectrum, applyFilter, nextPowerOf2, type FilterKind } from './fft';

function generateTestImage(): string {
  const c = document.createElement('canvas');
  const s = 64; c.width = s; c.height = s;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, s, s);
  ctx.fillStyle = '#ddd';
  ctx.fillRect(8, 8, 20, 20); ctx.fillRect(36, 36, 16, 16);
  ctx.beginPath(); ctx.arc(48, 16, 8, 0, Math.PI * 2); ctx.fill();
  ctx.fillRect(8, 40, 10, 12);
  return c.toDataURL();
}

type FilterType = 'low' | 'high' | 'band-pass' | 'band-stop';

export function FrequencySection() {
  const srcRef = useRef<HTMLImageElement>(null);
  const specRef = useRef<HTMLCanvasElement>(null);
  const filteredRef = useRef<HTMLCanvasElement>(null);
  const [filterType, setFilterType] = useState<FilterType>('low');
  const [filterKind, setFilterKind] = useState<FilterKind>('gaussian');
  const [cutoff, setCutoff] = useState(12);
  const [showCode, setShowCode] = useState(false);
  const [cutoff2, setCutoff2] = useState(24);
  const [view, setView] = useState<'filtered' | 'spectrum'>('filtered');

  useEffect(() => {
    if (srcRef.current) srcRef.current.src = generateTestImage();
  }, []);

  const process = useCallback(async () => {
    const img = srcRef.current;
    if (!img || !img.complete) return;
    const w = img.naturalWidth, h = img.naturalHeight;
    const fw = nextPowerOf2(w), fh = nextPowerOf2(h);

    // Read image
    const c = document.createElement('canvas');
    c.width = fw; c.height = fh;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, fw, fh);
    const gray = rgbaToGrayscale(imageData.data, fw, fh);

    // FFT
    const complex = pixelsToComplex(gray, fw, fh);
    const freq = fft2d(complex);
    const spectrum = magnitudeSpectrum(freq, fw, fh);

    // Draw spectrum
    const sCanvas = specRef.current;
    if (sCanvas) {
      sCanvas.width = fw; sCanvas.height = fh;
      const sCtx = sCanvas.getContext('2d')!;
      const sRGBA = grayscaleToRGBA(spectrum, fw, fh);
      sCtx.putImageData(new ImageData(sRGBA, fw, fh), 0, 0);
    }

    // Filter
    const filtered = applyFilter(freq, cutoff, filterType, filterKind, cutoff2);
    const filteredTime = ifft2d(filtered);
    const result = new Float64Array(fw * fh);
    for (let y = 0; y < fh; y++)
      for (let x = 0; x < fw; x++)
        result[y * fw + x] = Math.max(0, Math.min(255, filteredTime[y][x][0]));

    // Draw filtered
    const fCanvas = filteredRef.current;
    if (fCanvas) {
      fCanvas.width = fw; fCanvas.height = fh;
      const fCtx = fCanvas.getContext('2d')!;
      const fRGBA = grayscaleToRGBA(result, fw, fh);
      fCtx.putImageData(new ImageData(fRGBA, fw, fh), 0, 0);
    }
  }, [filterType, cutoff, cutoff2, filterKind]);

  useEffect(() => {
    const img = srcRef.current;
    if (!img) return;
    const onLoad = () => process();
    img.onload = onLoad;
    img.src = generateTestImage();
    if (img.complete && img.naturalWidth) process();
    return () => { img.onload = null; };
  }, [process]);

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Filter:</span>
        <button className={`btn btn-sm${filterType === 'low' ? ' btn-primary' : ''}`} onClick={() => setFilterType('low')}>Low-pass</button>
        <button className={`btn btn-sm${filterType === 'high' ? ' btn-primary' : ''}`} onClick={() => setFilterType('high')}>High-pass</button>
        <button className={`btn btn-sm${filterType === 'band-pass' ? ' btn-primary' : ''}`} onClick={() => setFilterType('band-pass')}>Band-pass</button>
        <button className={`btn btn-sm${filterType === 'band-stop' ? ' btn-primary' : ''}`} onClick={() => setFilterType('band-stop')}>Band-stop</button>
        <span className="demo-label" style={{ marginLeft: '1rem' }}>Type:</span>
        <button className={`btn btn-sm${filterKind === 'gaussian' ? ' btn-primary' : ''}`} onClick={() => setFilterKind('gaussian')}>Gaussian</button>
        <button className={`btn btn-sm${filterKind === 'ideal' ? ' btn-primary' : ''}`} onClick={() => setFilterKind('ideal')}>Ideal</button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: "0.5rem" }}>💻 Show code</button>
        <span className="demo-label" style={{ marginLeft: '1rem' }}>View:</span>
        <button className={`btn btn-sm${view === 'filtered' ? ' btn-primary' : ''}`} onClick={() => setView('filtered')}>Filtered</button>
        <button className={`btn btn-sm${view === 'spectrum' ? ' btn-primary' : ''}`} onClick={() => setView('spectrum')}>Spectrum</button>
      </div>

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <span className="demo-slider-label">Cutoff 1: <strong>{cutoff}</strong> px</span>
          <input type="range" min={2} max={31} value={cutoff} onChange={e => setCutoff(Number(e.target.value))} className="demo-slider" />
        </div>
        {(filterType === 'band-pass' || filterType === 'band-stop') && (
          <div className="demo-slider-row">
            <span className="demo-slider-label">Cutoff 2: <strong>{cutoff2}</strong> px</span>
            <input type="range" min={2} max={31} value={cutoff2} onChange={e => setCutoff2(Number(e.target.value))} className="demo-slider" />
          </div>
        )}
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <img ref={srcRef} className="color-img" alt="original" />
        </div>
        <div className="demo-panel">
          <h3>Spectrum</h3>
          <canvas ref={specRef} className="color-img" />
        </div>
        {view === 'filtered' && (
          <div className="demo-panel">
            <h3>{filterType === 'low' ? 'Low-pass (blur)' : filterType === 'high' ? 'High-pass (edges)' : filterType === 'band-pass' ? 'Band-pass' : 'Band-stop'}</h3>
            <canvas ref={filteredRef} className="color-img" />
          </div>
        )}
      </div>

      <div className="demo-explanation">
        <h4>Frequency Domain</h4>
        <p>The <strong>Fourier transform</strong> decomposes an image into sine/cosine waves of different frequencies.
        Each pixel in the spectrum represents the strength of a particular frequency and orientation.</p>
        <p><strong>Low-pass filter:</strong> removes high frequencies (sharp edges, noise), leaving only smooth variations → blur.</p>
        <p><strong>High-pass filter:</strong> removes low frequencies (smooth areas), leaving only sharp transitions → edges.</p>
        <p>The cutoff radius controls how much of the spectrum is kept. A smaller cutoff with low-pass = more blur.
        A larger cutoff with high-pass = fewer edges preserved.</p>
        <p>The bright centre of the spectrum represents low frequencies (the overall brightness/shape).
        The spokes outward represent edges at different orientations.</p>
      </div>
    </div>
      {showCode && <CodeViewer code={fftCode} title="fft.ts" onClose={() => setShowCode(false)} />}
    </>
  );
}
