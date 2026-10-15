import { useDeferredValue, useMemo, useState } from 'react';
import deblurCode from './deblur.ts?raw';
import pythonCode from './deblur.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { scene } from '../../lib/demoImages';
import { grayToImageData, rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { hotColor } from '../hough/hough';
import { psnr } from '../denoising/denoise';
import { nextPowerOf2 } from '../frequency/fft';
import { blurImage, makePsf, psfSpectrum, responseImage, richardsonLucy, wiener, type PsfKind } from './deblur';
import './deblur.css';

const PSF_SIZE = 21;
const DISPLAY = 200;

/** The image in a power-of-two square (needed by the FFT): centred, borders repeated outwards. */
function toSquare(gray: Float64Array, w: number, h: number) {
  const n = nextPowerOf2(Math.max(w, h));
  const ox = (n - w) >> 1, oy = (n - h) >> 1;
  const out = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const sx = Math.min(w - 1, Math.max(0, x - ox)), sy = Math.min(h - 1, Math.max(0, y - oy));
    out[y * n + x] = gray[sy * w + sx];
  }
  return { pixels: out, n };
}

const PARAM: Record<PsfKind, { label: string; min: number; max: number; step: number; initial: number }> = {
  motion: { label: 'Length (px)', min: 2, max: 20, step: 1, initial: 11 },
  gaussian: { label: 'σ (px)', min: 0.5, max: 5, step: 0.25, initial: 2 },
  defocus: { label: 'Radius (px)', min: 1, max: 9, step: 0.5, initial: 4 },
};

export function DeblurSection() {
  const [source, setSource] = useState<ImageSource>(() => scene(256));
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [kind, setKind] = useState<PsfKind>('motion');
  const [param, setParam] = useState(PARAM.motion.initial);
  const [angle, setAngle] = useState(20);
  const [noise, setNoise] = useState(2);
  const [logK, setLogK] = useState(-2);
  const [iterations, setIterations] = useState(20);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, 256);
  const K = 10 ** logK;

  const setup = useMemo(() => {
    if (!image) return null;
    const { pixels, n } = toSquare(image.gray, image.width, image.height);
    const psf = makePsf(kind, PSF_SIZE, param, angle);
    const H = psfSpectrum(psf, PSF_SIZE, n, n);
    const blurred = blurImage(pixels, H, noise, seededRandom(7));
    return { sharp: pixels, n, psf, H, blurred };
  }, [image, kind, param, angle, noise]);

  const inverse = useMemo(() => (setup ? wiener(setup.blurred, setup.H, 0) : null), [setup]);
  const wienerOut = useMemo(() => (setup ? wiener(setup.blurred, setup.H, K) : null), [setup, K]);
  // Richardson–Lucy runs two FFT convolutions per iteration: let the sliders stay responsive.
  const deferredSetup = useDeferredValue(setup);
  const deferredIterations = useDeferredValue(iterations);
  const rl = useMemo(() => (deferredSetup ? richardsonLucy(deferredSetup.blurred, deferredSetup.H, deferredIterations) : null), [deferredSetup, deferredIterations]);

  const show = (px: Float64Array | null) => (px && setup ? grayToImageData(px, setup.n, setup.n) : null);
  const score = (px: Float64Array | null) => (px && setup ? `PSNR ${psnr(setup.sharp, px.map(v => Math.min(255, Math.max(0, v)))).toFixed(1)} dB` : '');
  const psfImage = useMemo(() => {
    if (!setup) return null;
    const max = Math.max(...setup.psf);
    return rgbImage(PSF_SIZE, PSF_SIZE, i => hotColor(setup.psf[i] / max));
  }, [setup]);
  const response = useMemo(() => {
    if (!setup) return null;
    const r = responseImage(setup.H);
    return rgbImage(setup.n, setup.n, i => hotColor(Math.sqrt(r[i])));
  }, [setup]);
  const p = PARAM[kind];
  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Blur:</span>
        {(['motion', 'gaussian', 'defocus'] as const).map(k => (
          <button key={k} type="button" aria-pressed={kind === k} className={`btn btn-sm${kind === k ? ' btn-primary' : ''}`}
            onClick={() => { setKind(k); setParam(PARAM[k].initial); }}>
            {k === 'motion' ? 'Camera shake (motion)' : k === 'gaussian' ? 'Gaussian' : 'Out of focus (disc)'}
          </button>
        ))}
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setUploadError(null); }} onError={setUploadError} />
        <span className="ml-auto">
          <ShowCodeButton code={deblurCode} title="deblur/deblur.ts" python={{ code: pythonCode, title: 'deblur/deblur.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <label className="dl-slider">{p.label} <strong>{param}</strong>
          <input type="range" min={p.min} max={p.max} step={p.step} value={param} onChange={e => setParam(Number(e.target.value))} />
        </label>
        {kind === 'motion' && (
          <label className="dl-slider">Angle <strong>{angle}°</strong>
            <input type="range" min={0} max={180} value={angle} onChange={e => setAngle(Number(e.target.value))} />
          </label>
        )}
        <label className="dl-slider">Noise σ <strong>{noise}</strong>
          <input type="range" min={0} max={20} step={0.5} value={noise} onChange={e => setNoise(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Wiener K <strong>{K < 0.001 ? K.toExponential(0) : K.toFixed(3)}</strong>
          <input type="range" min={-5} max={0} step={0.1} value={logK} onChange={e => setLogK(Number(e.target.value))} />
        </label>
        <label className="dl-slider">R–L iterations <strong>{iterations}</strong>
          <input type="range" min={1} max={60} value={iterations} onChange={e => setIterations(Number(e.target.value))} />
        </label>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {setup && (
        <div className="deblur-grid">
          <Panel title="Sharp original" image={show(setup.sharp)} />
          <div className="demo-panel deblur-psf">
            <h3>Blur kernel (PSF)</h3>
            <PixelCanvas image={psfImage} size={96} label="The point spread function" />
            <h3>Its frequency response |H|</h3>
            <PixelCanvas image={response} size={96} label="Magnitude of the blur's frequency response, zero frequency in the centre" />
            <p className="caption">Dark = frequencies the blur destroys.</p>
          </div>
          <Panel title={`Blurred${noise ? ' + noise' : ''}`} image={show(setup.blurred)} caption={score(setup.blurred)} />
          <Panel title="Inverse filter G / H" image={show(inverse)} caption={score(inverse)} />
          <Panel title={`Wiener, K = ${K < 0.001 ? K.toExponential(0) : K.toFixed(3)}`} image={show(wienerOut)} caption={score(wienerOut)} />
          <Panel title={`Richardson–Lucy, ${deferredIterations} iterations`} image={show(rl)} caption={score(rl)} />
        </div>
      )}

      <div className="demo-explanation">
        <h4>Why undoing a blur is hard</h4>
        <p>A blur replaces every point with the kernel's shape: the image is <strong>convolved</strong> with the point spread function.
        In the frequency domain (see the Frequency lesson) that is just a multiplication, G = H · F, so dividing by H should undo it.
        Set the noise to 0 and the <strong>inverse filter</strong> is nearly perfect.</p>
        <p>Add even a little noise and it explodes. The response |H| has frequencies where it is close to zero (the dark stripes and
        rings), and noise divided by almost nothing becomes enormous. The <strong>Wiener filter</strong> divides by |H|² + K instead:
        where the blur kept the signal it inverts it, where it destroyed it, it gives up gracefully. Larger K means less noise but
        less sharpening. <strong>Richardson–Lucy</strong> instead refines an estimate step by step, keeping it positive; run too many
        iterations on a noisy image and it starts sharpening the noise too.</p>
        <p className="formula"><code>F̂ = H* · G / (|H|² + K)</code></p>
        <p>This lesson uses circular convolution, so blur wraps around the image edges. Real photos need their borders handled, and real
        deblurring also has to estimate the PSF itself (blind deconvolution).</p>
      </div>
    </div>
  );
}

function Panel({ title, image, caption }: { title: string; image: ImageData | null; caption?: string }) {
  return (
    <div className="demo-panel">
      <h3>{title}</h3>
      <PixelCanvas image={image} size={DISPLAY} label={title} />
      {caption && <p className="caption mono">{caption}</p>}
    </div>
  );
}
