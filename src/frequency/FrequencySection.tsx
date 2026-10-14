import { useMemo, useState } from 'react';
import fftCode from './fft.ts?raw';
import { ShowCodeButton } from '../components/CodeViewer';
import { PixelCanvas } from '../components/PixelCanvas';
import { ImageUploader } from '../components/ImageUploader';
import { useImageData, type ImageSource } from '../hooks/useImageData';
import { makeCanvas } from '../lib/demoImages';
import { grayToImageData } from '../lib/image';
import { normaliseRange } from '../processors/utils';
import {
  applyMask, cropReal, fft2d, filterMask, ifft2d, magnitudeSpectrum, padToPowerOfTwo,
  type FilterKind, type FilterType,
} from './fft';

function testImage(): HTMLCanvasElement {
  return makeCanvas(64, 64, ctx => {
    ctx.fillStyle = '#222'; ctx.fillRect(0, 0, 64, 64);
    ctx.fillStyle = '#ddd';
    ctx.fillRect(8, 8, 20, 20); ctx.fillRect(36, 36, 16, 16);
    ctx.beginPath(); ctx.arc(48, 16, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(8, 40, 10, 12);
  });
}

const FILTERS: { key: FilterType; label: string; heading: string }[] = [
  { key: 'low', label: 'Low-pass', heading: 'Low-pass (blur)' },
  { key: 'high', label: 'High-pass', heading: 'High-pass (edges)' },
  { key: 'band-pass', label: 'Band-pass', heading: 'Band-pass' },
  { key: 'band-stop', label: 'Band-stop', heading: 'Band-stop' },
];

type SpectrumView = 'original' | 'mask' | 'filtered';
const SPECTRUM_VIEWS: { key: SpectrumView; label: string }[] = [
  { key: 'original', label: 'Image spectrum' },
  { key: 'mask', label: 'Filter' },
  { key: 'filtered', label: 'Filtered spectrum' },
];

const DISPLAY = 240;

export function FrequencySection() {
  const [filterType, setFilterType] = useState<FilterType>('low');
  const [filterKind, setFilterKind] = useState<FilterKind>('gaussian');
  const [cutoff, setCutoff] = useState(8);
  const [cutoff2, setCutoff2] = useState(20);
  const [spectrumView, setSpectrumView] = useState<SpectrumView>('original');
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(testImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 128);

  // The FFT only depends on the image; filters reuse it.
  const transform = useMemo(() => {
    if (!image) return null;
    const padded = padToPowerOfTwo(image.gray, image.width, image.height);
    const freq = fft2d(padded);
    return { freq, spectrum: magnitudeSpectrum(freq) };
  }, [image]);

  const band = filterType === 'band-pass' || filterType === 'band-stop';
  const maxRadius = transform ? Math.min(transform.freq.width, transform.freq.height) / 2 : 32;
  // Keep the radii valid whatever the previous filter left them at (inner < outer ≤ max).
  const r2 = Math.min(Math.max(cutoff2, 2), maxRadius);
  const r1 = Math.min(cutoff, band ? r2 - 1 : maxRadius);

  const filtered = useMemo(() => {
    if (!image || !transform) return null;
    const { freq } = transform;
    const mask = filterMask(freq.width, freq.height, filterType, filterKind, r1, band ? r2 : r1);
    const result = applyMask(freq, mask);
    const spatial = cropReal(ifft2d(result), image.width, image.height);
    // Filters that remove the DC term produce values around 0 (positive and negative):
    // stretch them so 0 is mid-gray instead of clipping the negative half to black.
    const removesDc = filterType === 'high' || filterType === 'band-pass';
    return {
      mask: mask.map(g => g * 255),
      spectrum: magnitudeSpectrum(result),
      output: removesDc ? normaliseRange(spatial) : spatial,
      removesDc,
    };
  }, [image, transform, filterType, filterKind, r1, r2, band]);

  const spectrumImage = useMemo(() => {
    if (!transform || !filtered) return null;
    const { width, height } = transform.freq;
    const data = spectrumView === 'original' ? transform.spectrum : spectrumView === 'mask' ? filtered.mask : filtered.spectrum;
    return grayToImageData(data, width, height);
  }, [transform, filtered, spectrumView]);

  const outputImage = useMemo(
    () => (image && filtered ? grayToImageData(filtered.output, image.width, image.height) : null),
    [image, filtered],
  );

  const radiusLabel = filterKind === 'gaussian' ? 'σ' : 'Radius';
  const heading = FILTERS.find(f => f.key === filterType)!.heading;
  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Filter:</span>
        {FILTERS.map(f => (
          <button key={f.key} type="button" aria-pressed={filterType === f.key}
            className={`btn btn-sm${filterType === f.key ? ' btn-primary' : ''}`} onClick={() => setFilterType(f.key)}>
            {f.label}
          </button>
        ))}
        <span className="demo-label">Type:</span>
        <button type="button" aria-pressed={filterKind === 'gaussian'} className={`btn btn-sm${filterKind === 'gaussian' ? ' btn-primary' : ''}`} onClick={() => setFilterKind('gaussian')}>Gaussian</button>
        <button type="button" aria-pressed={filterKind === 'ideal'} className={`btn btn-sm${filterKind === 'ideal' ? ' btn-primary' : ''}`} onClick={() => setFilterKind('ideal')}>Ideal</button>
        <ImageUploader label="📁 Upload" onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(testImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={fftCode} title="frequency/fft.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="freq-r1">
            {band ? `Inner ${radiusLabel.toLowerCase()}` : radiusLabel}: <strong>{r1}</strong> {filterKind === 'ideal' ? 'bins' : ''}
          </label>
          <input id="freq-r1" type="range" min={1} max={band ? r2 - 1 : maxRadius} value={r1}
            onChange={e => setCutoff(Number(e.target.value))} className="demo-slider" />
        </div>
        {band && (
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="freq-r2">
              Outer {radiusLabel.toLowerCase()}: <strong>{r2}</strong> {filterKind === 'ideal' ? 'bins' : ''}
            </label>
            <input id="freq-r2" type="range" min={r1 + 1} max={maxRadius} value={r2}
              onChange={e => setCutoff2(Number(e.target.value))} className="demo-slider" />
          </div>
        )}
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Original image" />
        </div>
        <div className="demo-panel">
          <h3>Spectrum</h3>
          <div className="segmented" role="group" aria-label="Spectrum view">
            {SPECTRUM_VIEWS.map(v => (
              <button key={v.key} type="button" aria-pressed={spectrumView === v.key}
                className={`btn btn-sm${spectrumView === v.key ? ' btn-primary' : ''}`} onClick={() => setSpectrumView(v.key)}>
                {v.label}
              </button>
            ))}
          </div>
          <PixelCanvas image={spectrumImage} size={DISPLAY} label={SPECTRUM_VIEWS.find(v => v.key === spectrumView)!.label} />
          <p className="caption">Centre = low frequencies · log magnitude</p>
        </div>
        <div className="demo-panel">
          <h3>{heading}</h3>
          <PixelCanvas image={outputImage} size={DISPLAY} label={`${heading} result`} />
          {filtered?.removesDc && <p className="caption">Contrast-stretched: mid-gray = 0</p>}
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Frequency Domain</h4>
        <p>The <strong>Fourier transform</strong> decomposes an image into sine/cosine waves of different frequencies.
        Each point in the spectrum is one wave: its distance from the centre is the frequency, its direction the orientation, its brightness the strength.</p>
        <p><strong>Low-pass:</strong> keeps the centre (smooth variations) → blur. <strong>High-pass:</strong> keeps the outside (sharp transitions) → edges.
        Band filters keep or remove a ring between two radii. Switch the spectrum to <em>Filter</em> to see exactly which frequencies are kept (white) or removed (black).</p>
        <p><strong>Ideal</strong> filters cut sharply, which causes ringing (ripples next to edges); <strong>Gaussian</strong> filters roll off smoothly and avoid it.
        For Gaussian filters the slider sets σ of the Gaussian, for ideal filters the cut-off radius.</p>
        <p>The FFT needs power-of-two sizes, so images are padded with their mean value before the transform and cropped back afterwards.</p>
      </div>
    </div>
  );
}
