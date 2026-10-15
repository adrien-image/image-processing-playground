import { useMemo, useState } from 'react';
import { PixelCanvas } from '../components/PixelCanvas';
import { ImageUploader } from '../components/ImageUploader';
import { useImageData, type ImageSource } from '../hooks/useImageData';
import { clamp } from '../processors/utils';
import { splitRGB } from './plane';
import { antiAliasSigma, blurThenSubsample, crop, resize, subsample, type ResizeMethod } from './resample';
import { DEMO_SIZE, planesToImageData, zonePlateCanvas } from './view';
import { CodeMenu } from './CodeMenu';

const DISPLAY = 256;
const FACTORS = [2, 4, 8] as const;
const CROP = 24;
const ZOOM = 8;

const METHODS: { key: ResizeMethod; label: string; caption: string }[] = [
  { key: 'nearest', label: 'Nearest', caption: 'Copies the closest pixel → 8×8 blocks' },
  { key: 'bilinear', label: 'Bilinear', caption: 'Straight lines between 2×2 neighbours → smooth, soft, creased' },
  { key: 'bicubic', label: 'Bicubic', caption: 'Smooth cubic curves over 4×4 neighbours → smoother and sharper' },
];

export function ResamplingTab() {
  const [factor, setFactor] = useState<number>(4);
  const [source, setSource] = useState<ImageSource>(zonePlateCanvas);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // Crop centre as a fraction of the image size, so it survives a change of image.
  const [cropFx, setCropFx] = useState(0.625);
  const [cropFy, setCropFy] = useState(0.625);
  const { image, error: loadError } = useImageData(source, DEMO_SIZE);

  // Stage 1: R, G, B planes (only when the image changes).
  const planes = useMemo(() => (image ? splitRGB(image.imageData.data, image.width, image.height) : null), [image]);

  // Stage 2: the two downsampled versions (image or factor changes).
  const naive = useMemo(() => (planes ? planesToImageData(planes.map(p => subsample(p, factor))) : null), [planes, factor]);
  const filtered = useMemo(() => (planes ? planesToImageData(planes.map(p => blurThenSubsample(p, factor))) : null), [planes, factor]);

  // Stage 3: the enlarged crop (image or crop position changes).
  const w = image?.width ?? DEMO_SIZE, h = image?.height ?? DEMO_SIZE;
  const cw = Math.min(CROP, w), ch = Math.min(CROP, h);
  const x0 = clamp(Math.round(cropFx * w - cw / 2), 0, w - cw);
  const y0 = clamp(Math.round(cropFy * h - ch / 2), 0, h - ch);
  const enlarged = useMemo(() => {
    if (!planes) return null;
    const crops = planes.map(p => crop(p, x0, y0, cw, ch));
    return METHODS.map(m => planesToImageData(crops.map(c => resize(c, cw * ZOOM, ch * ZOOM, m.key))));
  }, [planes, x0, y0, cw, ch]);

  // No image (e.g. a failed upload): say so instead of describing the default size.
  const small = image ? `${Math.ceil(w / factor)}×${Math.ceil(h / factor)}` : '–';
  const error = uploadError ?? loadError;
  const pct = (v: number, of: number) => `${(v / of) * 100}%`;

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">Shrink by:</span>
        {FACTORS.map(f => (
          <button key={f} type="button" aria-pressed={factor === f}
            className={`btn btn-sm${factor === f ? ' btn-primary' : ''}`} onClick={() => setFactor(f)}>
            ÷{f}
          </button>
        ))}
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
        {!isDemo && (
          <button type="button" className="btn btn-sm" onClick={() => { setSource(zonePlateCanvas()); setIsDemo(true); setUploadError(null); }}>
            Zone plate
          </button>
        )}
        <CodeMenu initial="resample.ts" />
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel pyr-shrink-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Original image, with the enlarged crop outlined">
            <div className="pyr-crop-box" style={{ left: pct(x0, w), top: pct(y0, h), width: pct(cw, w), height: pct(ch, h) }} />
          </PixelCanvas>
          <p className="caption">{image ? `${w}×${h}` : '–'} · yellow box = crop enlarged below</p>
        </div>
        <div className="demo-panel pyr-shrink-panel">
          <h3>Naive ÷{factor}</h3>
          <PixelCanvas image={naive} size={DISPLAY} label={`Naive subsampling by ${factor}, enlarged`} placeholderAspect={w / h} />
          <p className="caption">
            Keep 1 pixel in {factor} along each axis (1 in {factor * factor} overall) → {small}:
            aliasing ({isDemo ? 'false rings' : 'false patterns'})
          </p>
        </div>
        <div className="demo-panel pyr-shrink-panel">
          <h3>Blur, then ÷{factor}</h3>
          <PixelCanvas image={filtered} size={DISPLAY} label={`Gaussian blur then subsampling by ${factor}, enlarged`} placeholderAspect={w / h} />
          <p className="caption">
            Gaussian σ = {antiAliasSigma(factor)} px first → {small}: {isDemo ? 'most fine rings fade to gray' : 'fine detail fades instead of aliasing'}
          </p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Why blur before shrinking? (Nyquist)</h4>
        <p>Keeping one pixel in {factor} along each axis samples the image every {factor} px. A sampled signal can only hold frequencies up to
        the <strong>Nyquist limit</strong>: half the sampling rate, here 1/(2·{factor}) = {1 / (2 * factor)} cycles/pixel
        — a pattern needs at least 2 samples per period. Finer patterns do not disappear: they <strong>fold back</strong> as false,
        coarser ones (aliasing). The zone plate's rings get finer away from the centre, so beyond the limit the naive result shows
        phantom ring centres (moiré).</p>
        <p>A <strong>low-pass filter first</strong> removes most of what the small image cannot represent. Fine rings then fade to
        the flat mean gray they average to, which is the honest answer at that resolution. A Gaussian is a <em>soft</em> filter,
        though: with σ = {antiAliasSigma(factor)} it still passes about 29 % at the new limit, so frequencies just above it leave
        faint false rings (look closely at ÷2, especially near the edges). A larger σ would remove them but blur the result more;
        anti-aliasing always trades a little aliasing against a little blur. Cameras do the same with an optical anti-aliasing
        filter, and image pyramids blur before every halving.</p>
      </div>

      <div className="demo-slider-section">
        <p className="panel-heading">Upscaling: {cw}×{ch} crop enlarged ×{ZOOM}</p>
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="pyr-crop-x">Crop centre x: <strong>{image ? x0 + cw / 2 : '–'}</strong> px</label>
          <input id="pyr-crop-x" type="range" className="demo-slider" min={cw / 2} max={w - cw / 2} value={x0 + cw / 2}
            disabled={!image || w <= cw} onChange={e => setCropFx(Number(e.target.value) / w)} />
        </div>
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="pyr-crop-y">Crop centre y: <strong>{image ? y0 + ch / 2 : '–'}</strong> px</label>
          <input id="pyr-crop-y" type="range" className="demo-slider" min={ch / 2} max={h - ch / 2} value={y0 + ch / 2}
            disabled={!image || h <= ch} onChange={e => setCropFy(Number(e.target.value) / h)} />
        </div>
      </div>

      <div className="demo-images">
        {METHODS.map((m, i) => (
          <div key={m.key} className="demo-panel pyr-zoom-panel">
            <h3>{m.label}</h3>
            <PixelCanvas image={enlarged?.[i] ?? null} size={CROP * ZOOM} label={`Crop enlarged ×${ZOOM} with ${m.label.toLowerCase()} interpolation`} />
            <p className="caption">{m.caption}</p>
          </div>
        ))}
      </div>

      <div className="demo-explanation">
        <h4>Interpolation</h4>
        <p>Enlarging must invent values between the known pixels. All three methods use <strong>pixel-centre alignment</strong>:
        output pixel <span className="mono">i</span> reads input position <span className="mono">(i + 0.5)·in/out − 0.5</span>, so the
        image edges line up at every scale.</p>
        <p><strong>Nearest</strong> copies the closest pixel (blocky). <strong>Bilinear</strong> blends the 2×2 neighbours with weights
        that fall linearly with distance: continuous, but its slope jumps at every original pixel, leaving creases.
        <strong> Bicubic</strong> draws a smooth cubic curve from the 4×4 neighbours (Keys kernel with a = −0.5, also called
        Catmull-Rom): the curve passes through the pixels, and the outer neighbours only set its slope, so there are no creases.
        It is smoother, slightly sharper, and can overshoot a little next to hard edges. None of them can recover detail that was
        never sampled.</p>
      </div>
    </>
  );
}
