import { useMemo, useState } from 'react';
import keypointsCode from './keypoints.ts?raw';
import pythonCode from './keypoints.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { createCanvas, imageDataToCanvas } from '../../lib/canvas';
import { makeCanvas } from '../../lib/demoImages';
import { rgbaToGrayscale } from '../processors/utils';
import { floatImage, signedImage } from '../../deep-learning/shared/render';
import { buildScaleSpace, detectKeypoints, repeatability, type Keypoint } from './keypoints';
import './keypoints.css';

const SIZE = 160;
const DISPLAY = 300;

/** Blobs of many sizes, light on dark and dark on light, plus edges and corners. */
function demoImage(): HTMLCanvasElement {
  return makeCanvas(SIZE, SIZE, ctx => {
    const bg = ctx.createLinearGradient(0, 0, SIZE, SIZE);
    bg.addColorStop(0, '#2b2f3a');
    bg.addColorStop(1, '#4a5060');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = '#e8e8e8';
    for (const [x, y, r] of [[22, 24, 3], [48, 22, 5], [84, 26, 8], [128, 32, 13], [30, 70, 10], [24, 120, 4], [40, 140, 6]]) {
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = '#d0d0d0';
    ctx.fillRect(62, 56, 52, 44);
    ctx.fillStyle = '#1a1c22';
    for (const [x, y, r] of [[76, 70, 4], [100, 84, 8]]) {
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = '#bfc4d0';
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2, r = i % 2 ? 9 : 20;
      ctx.lineTo(128 + r * Math.cos(a), 122 + r * Math.sin(a));
    }
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#f2f2f2';
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText('SIFT', 60, 140);
  });
}

/** The image zoomed by `zoom` about its centre; uncovered borders get the mean colour. */
function zoomGray(image: ImageData, zoom: number): Float64Array {
  const { width: w, height: h } = image;
  const { ctx } = createCanvas(w, h);
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < image.data.length; i += 4) { r += image.data[i]; g += image.data[i + 1]; b += image.data[i + 2]; }
  const n = w * h;
  ctx.fillStyle = `rgb(${r / n}, ${g / n}, ${b / n})`;
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = 'high';
  ctx.translate(w / 2, h / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-w / 2, -h / 2);
  ctx.drawImage(imageDataToCanvas(image), 0, 0);
  return rgbaToGrayscale(ctx.getImageData(0, 0, w, h).data, w, h);
}

function KeypointOverlay({ keypoints, width, height, showAngles, scale = 1 }: {
  keypoints: Keypoint[]; width: number; height: number; showAngles: boolean; scale?: number;
}) {
  return (
    <svg className="kp-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {keypoints.map((kp, i) => {
        const r = (kp.sigma * Math.SQRT2) / scale;
        const x = kp.x / scale, y = kp.y / scale;
        return (
          <g key={i} className={kp.response < 0 ? 'kp-light' : 'kp-dark'}>
            <circle cx={x} cy={y} r={r} />
            {showAngles && <line x1={x} y1={y} x2={x + r * Math.cos(kp.angle)} y2={y + r * Math.sin(kp.angle)} />}
          </g>
        );
      })}
    </svg>
  );
}

export function KeypointsSection() {
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1.6);
  const [contrast, setContrast] = useState(0.02);
  const [edgeRatio, setEdgeRatio] = useState(10);
  const [maxShown, setMaxShown] = useState(40);
  const [showAngles, setShowAngles] = useState(true);
  const [octave, setOctave] = useState(0);
  const [level, setLevel] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, SIZE);

  const original = useMemo(() => {
    if (!image) return null;
    const space = buildScaleSpace(image.gray, image.width, image.height);
    return { space, keypoints: detectKeypoints(space, contrast, edgeRatio) };
  }, [image, contrast, edgeRatio]);

  const zoomed = useMemo(() => {
    if (!image) return null;
    const gray = zoomGray(image.imageData, zoom);
    const keypoints = detectKeypoints(buildScaleSpace(gray, image.width, image.height), contrast, edgeRatio);
    return { keypoints, display: floatImage(gray.map(v => v / 255), image.width, image.height) };
  }, [image, zoom, contrast, edgeRatio]);

  const stats = useMemo(() => {
    if (!image || !original || !zoomed) return null;
    const cx = image.width / 2, cy = image.height / 2;
    const toZoomed = (x: number, y: number): [number, number] => [cx + zoom * (x - cx), cy + zoom * (y - cy)];
    const shown = original.keypoints.slice(0, maxShown);
    return { repeat: repeatability(shown, zoomed.keypoints, toZoomed, zoom, image.width, image.height) };
  }, [image, original, zoomed, zoom, maxShown]);

  const oct = original?.space.octaves[Math.min(octave, original.space.octaves.length - 1)] ?? null;
  const levelImages = useMemo(() => {
    if (!oct) return null;
    const l = Math.min(level, oct.dogs.length - 1);
    return {
      gaussian: floatImage(oct.gaussians[l], oct.width, oct.height),
      dog: signedImage(oct.dogs[l], oct.width, oct.height),
      sigma: oct.sigmas[l],
      layer: l,
    };
  }, [oct, level]);

  const error = uploadError ?? loadError;
  const shownOriginal = original?.keypoints.slice(0, maxShown) ?? [];
  const shownZoomed = zoomed?.keypoints.slice(0, Math.round(maxShown * zoom * zoom)) ?? [];
  const levelKeypoints = original && levelImages ? original.keypoints.filter(k => k.octave === octave && k.layer === levelImages.layer) : [];

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Keypoints with a size: the same blob is found again when the image is zoomed</span>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={keypointsCode} title="keypoints/keypoints.ts" python={{ code: pythonCode, title: 'keypoints/keypoints.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Zoom <strong>×{zoom.toFixed(2)}</strong>
            <input type="range" min={0.5} max={2.5} step={0.05} value={zoom} onChange={e => setZoom(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Contrast threshold <strong>{contrast.toFixed(3)}</strong>
            <input type="range" min={0.005} max={0.08} step={0.005} value={contrast} onChange={e => setContrast(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Edge ratio r <strong>{edgeRatio}</strong>
            <input type="range" min={2} max={30} step={1} value={edgeRatio} onChange={e => setEdgeRatio(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Strongest <strong>{maxShown}</strong>
            <input type="range" min={5} max={150} step={5} value={maxShown} onChange={e => setMaxShown(Number(e.target.value))} />
          </label>
          <label className="dl-slider"><input type="checkbox" checked={showAngles} onChange={e => setShowAngles(e.target.checked)} /> Orientation</label>
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original · {original?.keypoints.length ?? 0} keypoints</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Image with its strongest keypoints as circles of radius σ√2">
            {image && <KeypointOverlay keypoints={shownOriginal} width={image.width} height={image.height} showAngles={showAngles} />}
          </PixelCanvas>
          <p className="caption"><span className="kp-key-light">○ light blob</span> · <span className="kp-key-dark">○ dark blob</span> · radius σ√2</p>
        </div>
        <div className="demo-panel">
          <h3>Zoomed ×{zoom.toFixed(2)} · {zoomed?.keypoints.length ?? 0} keypoints</h3>
          <PixelCanvas image={zoomed?.display ?? null}
            size={DISPLAY} label="The zoomed image with its keypoints">
            {image && <KeypointOverlay keypoints={shownZoomed} width={image.width} height={image.height} showAngles={showAngles} />}
          </PixelCanvas>
          {stats && <p className="caption"><strong>{Math.round(stats.repeat * 100)}%</strong> of the {maxShown} strongest original keypoints found again, at σ × {zoom.toFixed(2)}</p>}
        </div>
      </div>

      {original && oct && levelImages && (
        <div className="demo-panel">
          <h3>Inside the scale space</h3>
          <div className="demo-controls kp-space-controls">
            <span className="demo-label">Octave:</span>
            {original.space.octaves.map((o, i) => (
              <button key={i} type="button" aria-pressed={octave === i} className={`btn btn-sm${octave === i ? ' btn-primary' : ''}`}
                onClick={() => setOctave(i)}>{i} · {o.width}×{o.height}</button>
            ))}
            <label className="dl-slider">DoG layer <strong>{levelImages.layer}</strong>
              <input type="range" min={0} max={oct.dogs.length - 1} value={levelImages.layer} onChange={e => setLevel(Number(e.target.value))} />
            </label>
          </div>
          <div className="demo-images">
            <div>
              <PixelCanvas image={levelImages.gaussian} size={240} label={`Gaussian level, σ ${levelImages.sigma.toFixed(1)} pixels`} />
              <p className="caption">Gaussian, σ = {levelImages.sigma.toFixed(2)} px</p>
            </div>
            <div>
              <PixelCanvas image={levelImages.dog} size={240} label="Difference of Gaussians: blue negative, red positive">
                <KeypointOverlay keypoints={levelKeypoints} width={oct.width} height={oct.height} showAngles={false} scale={oct.step} />
              </PixelCanvas>
              <p className="caption">DoG = next level − this one · {levelKeypoints.length} extrema kept here
                {levelImages.layer === 0 || levelImages.layer === oct.dogs.length - 1 ? ' (edge layer: no neighbour on one side)' : ''}</p>
            </div>
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Why a keypoint needs a size</h4>
        <p>Harris corners (the Matching lesson) are found at one fixed scale: zoom in and a corner becomes a soft curve that
        the detector no longer sees. SIFT instead looks for <strong>blobs at every scale</strong>. It blurs the image with
        Gaussians of σ = 1.6, 1.6·k, 1.6·k², … (k = 2<sup>1/3</sup>) and subtracts neighbouring levels:</p>
        <p className="formula"><code>DoG(x, y, σ) = G(kσ) ⋆ I − G(σ) ⋆ I ≈ (k − 1) · σ² ∇²G ⋆ I</code></p>
        <p>The σ² factor makes the response comparable across scales, so a blob gives its strongest response at the σ that
        matches its size. A keypoint is a DoG value larger or smaller than its 8 neighbours in the same layer and the 9 in
        each layer above and below: an extremum in space <em>and</em> scale.</p>
        <p>Two tests remove unreliable points: a <strong>contrast threshold</strong> drops weak responses, and the
        <strong> edge test</strong> drops points whose DoG is a long ridge (an edge), whose position along the edge is
        undefined: it keeps a point only if the ratio of its principal curvatures is below r. Each octave halves the image,
        so a 40-pixel blob costs as much as a 5-pixel one. The orientation, from the strongest local gradient, is what makes
        the descriptor built around the keypoint rotation invariant.</p>
        <p>Zoom the image: the circles grow with the blobs, and most strong keypoints are found again at σ × zoom. Real SIFT
        also refines each point with a 3-D quadratic fit and doubles the image first to find the smallest blobs.</p>
      </div>
    </div>
  );
}
