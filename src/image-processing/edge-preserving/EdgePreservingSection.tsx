import { useMemo, useState, type MouseEvent } from 'react';
import filtersCode from './edgePreserving.ts?raw';
import pythonCode from './edge_preserving.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { grayToImageData } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { heatImage } from '../../deep-learning/shared/render';
import { addGaussianNoise, gaussianFilter, psnr } from '../denoising/denoise';
import { bilateralFilter, bilateralWeights, nlmWeights, nonLocalMeans, type NlmOptions } from './edgePreserving';
import './edge-preserving.css';

const SIZE = 112;
const DISPLAY = 210;
const BILATERAL_RADIUS = 3;

/** Smooth shading, hard edges and a repeated texture: each filter fails somewhere else. */
function demoImage(): HTMLCanvasElement {
  return makeCanvas(SIZE, SIZE, ctx => {
    const g = ctx.createLinearGradient(0, 0, SIZE, 0);
    g.addColorStop(0, '#303030');
    g.addColorStop(1, '#909090');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = '#e0e0e0';
    ctx.beginPath(); ctx.arc(32, 34, 20, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#141414';
    ctx.fillRect(62, 14, 36, 40);
    // Stripes: a texture that repeats, which non-local means can average with its copies.
    for (let x = 8; x < 104; x += 8) {
      ctx.fillStyle = '#d8d8d8';
      ctx.fillRect(x, 66, 4, 38);
    }
  });
}

const fmtDb = (v: number) => (Number.isFinite(v) ? `${v.toFixed(1)} dB` : '∞');

export function EdgePreservingSection() {
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [sigma, setSigma] = useState(20);
  const [seed, setSeed] = useState(1);
  const [sigmaSpace, setSigmaSpace] = useState(2);
  const [sigmaRange, setSigmaRange] = useState(40);
  const [hFactor, setHFactor] = useState(0.8);
  const [patchRadius, setPatchRadius] = useState(1);
  const [searchRadius, setSearchRadius] = useState(5);
  const [probe, setProbe] = useState({ x: 30, y: 20 });
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, SIZE);

  const noisy = useMemo(() => (image ? addGaussianNoise(image.gray, sigma, seededRandom(seed)) : null), [image, sigma, seed]);
  const nlmOptions: NlmOptions = useMemo(() => ({ patchRadius, searchRadius, sigma, h: hFactor * sigma }), [patchRadius, searchRadius, sigma, hFactor]);

  const gaussian = useMemo(() => (image && noisy ? gaussianFilter(noisy, image.width, image.height, BILATERAL_RADIUS) : null), [image, noisy]);
  const bilateral = useMemo(
    () => (image && noisy ? bilateralFilter(noisy, image.width, image.height, BILATERAL_RADIUS, sigmaSpace, sigmaRange) : null),
    [image, noisy, sigmaSpace, sigmaRange],
  );
  const nlm = useMemo(() => (image && noisy ? nonLocalMeans(noisy, image.width, image.height, nlmOptions) : null), [image, noisy, nlmOptions]);

  const weights = useMemo(() => {
    if (!image || !noisy) return null;
    const x = Math.min(probe.x, image.width - 1), y = Math.min(probe.y, image.height - 1);
    const b = 2 * BILATERAL_RADIUS + 1, n = 2 * searchRadius + 1;
    return {
      bilateral: heatImage(bilateralWeights(noisy, image.width, image.height, x, y, BILATERAL_RADIUS, sigmaSpace, sigmaRange), b, b),
      nlm: heatImage(nlmWeights(noisy, image.width, image.height, x, y, nlmOptions), n, n),
      x, y,
    };
  }, [image, noisy, probe, sigmaSpace, sigmaRange, nlmOptions, searchRadius]);

  const pick = (e: MouseEvent<HTMLDivElement>) => {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setProbe({
      x: Math.floor(((e.clientX - rect.left) / rect.width) * image.width),
      y: Math.floor(((e.clientY - rect.top) / rect.height) * image.height),
    });
  };

  const error = uploadError ?? loadError;
  const panel = (title: string, pixels: Float64Array | null, label: string) => (
    <div className="demo-panel">
      <h3>{title}</h3>
      <PixelCanvas image={image && pixels ? grayToImageData(pixels, image.width, image.height) : null} size={DISPLAY} label={label} />
      {image && pixels && <p className="caption">PSNR {fmtDb(psnr(image.gray, pixels))}</p>}
    </div>
  );

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Smooth the noise, keep the edges</span>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 New noise</button>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={filtersCode} title="edge-preserving/edgePreserving.ts" python={{ code: pythonCode, title: 'edge-preserving/edge_preserving.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Noise σ <strong>{sigma}</strong>
            <input type="range" min={5} max={40} value={sigma} onChange={e => setSigma(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Bilateral σs <strong>{sigmaSpace.toFixed(1)}</strong>
            <input type="range" min={0.5} max={4} step={0.1} value={sigmaSpace} onChange={e => setSigmaSpace(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Bilateral σr <strong>{sigmaRange}</strong>
            <input type="range" min={5} max={150} step={5} value={sigmaRange} onChange={e => setSigmaRange(Number(e.target.value))} />
          </label>
          <label className="dl-slider">NLM h <strong>{hFactor.toFixed(1)}σ</strong>
            <input type="range" min={0.2} max={2} step={0.1} value={hFactor} onChange={e => setHFactor(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Patch <strong>{2 * patchRadius + 1}×{2 * patchRadius + 1}</strong>
            <input type="range" min={1} max={2} value={patchRadius} onChange={e => setPatchRadius(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Search <strong>{2 * searchRadius + 1}×{2 * searchRadius + 1}</strong>
            <input type="range" min={2} max={7} value={searchRadius} onChange={e => setSearchRadius(Number(e.target.value))} />
          </label>
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Noisy · click a pixel</h3>
          <div className="ep-picker" onClick={pick} role="presentation">
            <PixelCanvas image={image && noisy ? grayToImageData(noisy, image.width, image.height) : null} size={DISPLAY} label="Noisy image; click to inspect the filters' weights at a pixel">
              {weights && image && (
                <svg className="ep-overlay" viewBox={`0 0 ${image.width} ${image.height}`} preserveAspectRatio="none" aria-hidden="true">
                  <rect className="ep-search" x={weights.x - searchRadius} y={weights.y - searchRadius} width={2 * searchRadius + 1} height={2 * searchRadius + 1} />
                  <rect className="ep-probe" x={weights.x} y={weights.y} width={1} height={1} />
                </svg>
              )}
            </PixelCanvas>
          </div>
          {image && noisy && <p className="caption">PSNR {fmtDb(psnr(image.gray, noisy))}</p>}
        </div>
        {panel(`Gaussian ${2 * BILATERAL_RADIUS + 1}×${2 * BILATERAL_RADIUS + 1}`, gaussian, 'Gaussian-filtered image')}
        {panel('Bilateral', bilateral, 'Bilateral-filtered image')}
        {panel('Non-local means', nlm, 'Non-local means result')}
      </div>

      {weights && (
        <div className="demo-images">
          <div className="demo-panel">
            <h3>Bilateral weights at ({weights.x}, {weights.y})</h3>
            <PixelCanvas image={weights.bilateral} size={150} label="Bilateral filter weights in the 7×7 window" />
            <p className="caption">7×7 window · bright = large weight</p>
          </div>
          <div className="demo-panel">
            <h3>Non-local means weights</h3>
            <PixelCanvas image={weights.nlm} size={150} label="Non-local means weights in the search window" />
            <p className="caption">{2 * searchRadius + 1}×{2 * searchRadius + 1} search window (the box on the noisy image)</p>
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Average only with what looks alike</h4>
        <p>A Gaussian blur gives each neighbour a weight that depends on distance alone, so near an edge it averages the two
        sides and the edge turns into a ramp. The <strong>bilateral filter</strong> multiplies that weight by a second
        Gaussian of the <em>value</em> difference:</p>
        <p className="formula"><code>w(p, q) = exp(−‖p − q‖² / 2σs²) · exp(−(I(q) − I(p))² / 2σr²)</code></p>
        <p>Pixels across an edge differ by much more than σr, so they get no weight: click next to an edge and the weight map
        is cut in two. But it compares single, noisy pixels: σr must be well above the noise σ, or the filter keeps the noise
        too. With σr huge it is a Gaussian blur again.</p>
        <p><strong>Non-local means</strong> compares whole patches instead, which averages the noise out of the comparison,
        and looks further away. The weight of a candidate q is</p>
        <p className="formula"><code>w(p, q) = exp(−max(d²(P(p), P(q)) − 2σ², 0) / h²)</code></p>
        <p>where d² is the mean squared difference of the two patches and 2σ² is what noise alone puts between two copies of
        the same patch. Click a stripe: the weights light up the <em>other</em> stripes, because a repeated texture is
        averaged with its own copies. That is why non-local means usually wins on PSNR, at a much higher cost:
        every pixel compares (2S+1)² patches.</p>
      </div>
    </div>
  );
}
