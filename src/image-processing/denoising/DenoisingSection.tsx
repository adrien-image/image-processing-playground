import { useMemo, useState } from 'react';
import denoiseCode from './denoise.ts?raw';
import pythonCode from './denoise.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { grayToImageData } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { useText } from '../../lib/i18n';
import { TEXT } from './DenoisingSection.text';
import { addGaussianNoise, addSaltPepper, gaussianFilter, gaussianSigmaForRadius, medianFilter, psnr, type NoiseType } from './denoise';

function demoImage(): HTMLCanvasElement {
  return makeCanvas(128, 128, ctx => {
    ctx.fillStyle = '#222'; ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = '#ddd';
    ctx.fillRect(20, 20, 50, 50);
    ctx.fillRect(80, 30, 30, 40);
    ctx.beginPath(); ctx.arc(80, 90, 25, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#aaa';
    ctx.fillRect(15, 80, 25, 35);
  });
}

const DISPLAY = 220;
const fmtDb = (v: number) => (Number.isFinite(v) ? `${v.toFixed(1)} dB` : '∞');

export function DenoisingSection() {
  const t = useText(TEXT);
  const [noiseType, setNoiseType] = useState<NoiseType>('salt-pepper');
  const [density, setDensity] = useState(0.08);
  const [sigma, setSigma] = useState(25);
  const [radius, setRadius] = useState(1);
  const [seed, setSeed] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 256);

  // The noisy image depends only on the noise settings and seed — changing the
  // filter window compares both filters on exactly the same noise.
  const noisy = useMemo(() => {
    if (!image) return null;
    const random = seededRandom(seed);
    return noiseType === 'salt-pepper' ? addSaltPepper(image.gray, density, random) : addGaussianNoise(image.gray, sigma, random);
  }, [image, noiseType, density, sigma, seed]);

  const results = useMemo(() => {
    if (!image || !noisy) return null;
    const { width: w, height: h, gray } = image;
    const median = medianFilter(noisy, w, h, radius);
    const gaussian = gaussianFilter(noisy, w, h, radius);
    const show = (px: Float64Array) => grayToImageData(px, w, h);
    return {
      original: show(gray),
      noisy: { image: show(noisy), psnr: psnr(gray, noisy) },
      median: { image: show(median), psnr: psnr(gray, median) },
      gaussian: { image: show(gaussian), psnr: psnr(gray, gaussian) },
    };
  }, [image, noisy, radius]);

  const size = radius * 2 + 1;
  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">{t.noise}</span>
        <button type="button" aria-pressed={noiseType === 'salt-pepper'} className={`btn btn-sm${noiseType === 'salt-pepper' ? ' btn-primary' : ''}`} onClick={() => setNoiseType('salt-pepper')}>{t.saltPepper}</button>
        <button type="button" aria-pressed={noiseType === 'gaussian'} className={`btn btn-sm${noiseType === 'gaussian' ? ' btn-primary' : ''}`} onClick={() => setNoiseType('gaussian')}>{t.gaussian}</button>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>{t.newNoise}</button>
        <ImageUploader label={t.upload} onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>{t.demoImage}</button>}
        <span className="ml-auto">
          <ShowCodeButton code={denoiseCode} title="denoising/denoise.ts" python={{ code: pythonCode, title: 'denoising/denoise.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        {noiseType === 'salt-pepper' ? (
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="dn-density">{t.density} <strong>{Math.round(density * 100)}%</strong></label>
            <input id="dn-density" type="range" min={0.01} max={0.3} step={0.01} value={density}
              onChange={e => setDensity(Number(e.target.value))} className="demo-slider" />
          </div>
        ) : (
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="dn-sigma">{t.noiseSigma} <strong>{sigma}</strong></label>
            <input id="dn-sigma" type="range" min={2} max={60} value={sigma}
              onChange={e => setSigma(Number(e.target.value))} className="demo-slider" />
          </div>
        )}
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="dn-radius">{t.window} <strong>{size}×{size}</strong></label>
          <input id="dn-radius" type="range" min={1} max={3} value={radius}
            onChange={e => setRadius(Number(e.target.value))} className="demo-slider" />
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>{t.original}</h3>
          <PixelCanvas image={results?.original ?? null} size={DISPLAY} label={t.originalLabel} />
        </div>
        <div className="demo-panel">
          <h3>{t.noisy(noiseType)}</h3>
          <PixelCanvas image={results?.noisy.image ?? null} size={DISPLAY} label={t.noisyLabel} />
          {results && <p className="caption">PSNR {fmtDb(results.noisy.psnr)}</p>}
        </div>
        <div className="demo-panel">
          <h3>{t.median(size)}</h3>
          <PixelCanvas image={results?.median.image ?? null} size={DISPLAY} label={t.medianLabel} />
          {results && <p className="caption">PSNR {fmtDb(results.median.psnr)}</p>}
        </div>
        <div className="demo-panel">
          <h3>{t.gaussianTitle(size, gaussianSigmaForRadius(radius).toFixed(1))}</h3>
          <PixelCanvas image={results?.gaussian.image ?? null} size={DISPLAY} label={t.gaussianLabel} />
          {results && <p className="caption">PSNR {fmtDb(results.gaussian.psnr)}</p>}
        </div>
      </div>

      <div className="demo-explanation">{t.explanation}</div>
    </div>
  );
}
