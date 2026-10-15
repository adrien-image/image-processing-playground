import { useMemo, useState } from 'react';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { gaussianBlur, splitRGB, type Plane } from './plane';
import { alphaBlend, blendLevelLimit, blendLevels, gaussianPyramid, hardCut, reconstruct } from './pyramid';
import { resizeAntialiased } from './resample';
import { makeMask, type MaskShape } from './demos';
import { byLevel, DEMO_SIZE, planesToImageData, pyramidsPerChannel, seaCanvas, sunsetCanvas } from './view';
import { CodeMenu } from './CodeMenu';

const DISPLAY = 256;
const INPUT_DISPLAY = 160;
const LEVEL_DISPLAY = 96;

const MASKS: { key: MaskShape; label: string }[] = [
  { key: 'vertical', label: 'Vertical split' },
  { key: 'horizontal', label: 'Horizontal split' },
  { key: 'disc', label: 'Centred disc' },
];

/** Give a plane the target size (anti-aliased bilinear), or return it as is when it already fits. */
const fit = (p: Plane, width: number, height: number) => (p.width === width && p.height === height ? p : resizeAntialiased(p, width, height));

export function BlendingTab() {
  const [sourceA, setSourceA] = useState<ImageSource>(sunsetCanvas);
  const [sourceB, setSourceB] = useState<ImageSource>(seaCanvas);
  const [demos, setDemos] = useState(true);
  const [maskUpload, setMaskUpload] = useState<string | null>(null);
  const [maskShape, setMaskShape] = useState<MaskShape>('vertical');
  const [useMaskUpload, setUseMaskUpload] = useState(false);
  const [levels, setLevels] = useState(5);
  const [sigma, setSigma] = useState(4);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const { image: imageA, error: errorA } = useImageData(sourceA, DEMO_SIZE);
  const { image: imageB, error: errorB } = useImageData(sourceB, DEMO_SIZE);
  const { image: maskImage, error: errorMask } = useImageData(maskUpload, DEMO_SIZE);
  // An uploaded mask that fails to decode falls back to the last mask shape (and shows the error).
  const maskUploadFailed = useMaskUpload && errorMask !== null;
  const customMask = useMaskUpload && !maskUploadFailed;

  // Stage 1: input planes. A sets the size; B and an uploaded mask are resized to it.
  const planesA = useMemo(() => (imageA ? splitRGB(imageA.imageData.data, imageA.width, imageA.height) : null), [imageA]);
  const planesB = useMemo(() => {
    if (!imageA || !imageB) return null;
    return splitRGB(imageB.imageData.data, imageB.width, imageB.height).map(p => fit(p, imageA.width, imageA.height));
  }, [imageA, imageB]);
  const mask = useMemo(() => {
    if (!imageA) return null;
    if (!customMask) return makeMask(maskShape, imageA.width, imageA.height);
    if (!maskImage) return null; // still loading
    const gray: Plane = { data: maskImage.gray.map(v => v / 255), width: maskImage.width, height: maskImage.height };
    return fit(gray, imageA.width, imageA.height);
  }, [imageA, customMask, maskShape, maskImage]);

  // Stop while the coarsest level is still ≥ 16 px: deeper residuals leak A's colours into B (see blendLevelLimit).
  const levelLimit = imageA ? blendLevelLimit(imageA.width, imageA.height) : blendLevelLimit(DEMO_SIZE, DEMO_SIZE);
  const n = Math.min(levels, levelLimit);

  // Stage 2: the deepest pyramids, each only when its own input changes. Fewer levels are
  // a slice of these (see laplacianLevels), so the levels slider never rebuilds them.
  const pyrA = useMemo(() => (planesA ? pyramidsPerChannel(planesA, levelLimit) : null), [planesA, levelLimit]);
  const pyrB = useMemo(() => (planesB ? pyramidsPerChannel(planesB, levelLimit) : null), [planesB, levelLimit]);
  const maskPyramid = useMemo(() => (mask ? gaussianPyramid(mask, levelLimit) : null), [mask, levelLimit]);

  // Stage 3: blended bands Mᵢ·LAᵢ + (1 − Mᵢ)·LBᵢ, per channel, at full depth.
  const blendedBands = useMemo(
    () => (pyrA && pyrB && maskPyramid ? pyrA.laplacian.map((la, c) => blendLevels(la, pyrB.laplacian[c], maskPyramid)) : null),
    [pyrA, pyrB, maskPyramid],
  );

  // Stage 4 (levels): n levels = blended bands 0 … n−2 + the blended residual at level n−1; collapse.
  const multiband = useMemo(() => {
    if (!pyrA || !pyrB || !maskPyramid || !blendedBands) return null;
    const residual = pyrA.gaussian.map((ga, c) => alphaBlend(ga[n - 1], pyrB.gaussian[c][n - 1], maskPyramid[n - 1]));
    const channels = blendedBands.map((bands, c) => reconstruct([...bands.slice(0, n - 1), residual[c]]));
    return { residual: planesToImageData(residual), image: planesToImageData(channels) };
  }, [pyrA, pyrB, maskPyramid, blendedBands, n]);

  // Hard cut and feathering don't use pyramids.
  const hard = useMemo(
    () => (planesA && planesB && mask ? planesToImageData(planesA.map((a, c) => hardCut(a, planesB[c], mask))) : null),
    [planesA, planesB, mask],
  );
  const softMask = useMemo(() => (mask ? gaussianBlur(mask, sigma) : null), [mask, sigma]);
  const feathered = useMemo(
    () => (planesA && planesB && softMask ? planesToImageData(planesA.map((a, c) => alphaBlend(a, planesB[c], softMask))) : null),
    [planesA, planesB, softMask],
  );

  // Display-only images.
  const imageBShown = useMemo(() => (planesB ? planesToImageData(planesB) : null), [planesB]);
  const maskShown = useMemo(() => (mask ? planesToImageData([mask], 0, 255) : null), [mask]);
  const maskLevels = useMemo(() => maskPyramid?.map(m => planesToImageData([m], 0, 255)) ?? [], [maskPyramid]);
  const bandImages = useMemo(() => (blendedBands ? byLevel(blendedBands).map(ps => planesToImageData(ps, 128, 2)) : []), [blendedBands]);
  const bandsShown = multiband ? [...bandImages.slice(0, n - 1), multiband.residual] : [];

  // A failed mask upload only matters until another mask is chosen.
  const error = uploadError ?? errorA ?? errorB ?? (maskUploadFailed ? `Mask: ${errorMask}` : null);
  const aspect = imageA ? imageA.width / imageA.height : 1;
  const coarsest = imageA ? `${Math.ceil(imageA.width / 2 ** (levelLimit - 1))}×${Math.ceil(imageA.height / 2 ** (levelLimit - 1))}` : '–';
  // A mask that failed to decode is dropped as soon as the user moves on, so its error does too.
  const dropFailedMask = () => {
    if (!maskUploadFailed) return;
    setMaskUpload(null);
    setUseMaskUpload(false);
  };
  const resetDemos = () => {
    setSourceA(sunsetCanvas());
    setSourceB(seaCanvas());
    setDemos(true);
    setUploadError(null);
    dropFailedMask();
  };

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">Images:</span>
        <ImageUploader label="📁 Image A" onImage={src => { setSourceA(src); setDemos(false); setUploadError(null); dropFailedMask(); }} onError={setUploadError} />
        <ImageUploader label="📁 Image B" onImage={src => { setSourceB(src); setDemos(false); setUploadError(null); dropFailedMask(); }} onError={setUploadError} />
        {!demos && <button type="button" className="btn btn-sm" onClick={resetDemos}>Demo images</button>}
        <span className="demo-label">Mask:</span>
        {MASKS.map(m => {
          const active = !customMask && maskShape === m.key;
          return (
            <button key={m.key} type="button" aria-pressed={active} className={`btn btn-sm${active ? ' btn-primary' : ''}`}
              onClick={() => { setMaskShape(m.key); setUseMaskUpload(false); }}>
              {m.label}
            </button>
          );
        })}
        {maskUpload && !errorMask && (
          <button type="button" aria-pressed={customMask} className={`btn btn-sm${customMask ? ' btn-primary' : ''}`}
            onClick={() => setUseMaskUpload(true)}>
            Uploaded
          </button>
        )}
        <ImageUploader label="📁 Mask" onImage={src => { setMaskUpload(src); setUseMaskUpload(true); setUploadError(null); }} onError={setUploadError} />
        <CodeMenu initial="pyramid.ts" />
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="pyr-blend-levels">Pyramid levels: <strong>{imageA ? n : '–'}</strong></label>
          <input id="pyr-blend-levels" type="range" className="demo-slider" min={1} max={levelLimit} value={n}
            disabled={!imageA || levelLimit < 2} onChange={e => setLevels(Number(e.target.value))} />
        </div>
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="pyr-feather">Feather σ: <strong>{sigma}</strong> px</label>
          <input id="pyr-feather" type="range" className="demo-slider" min={1} max={32} value={sigma}
            onChange={e => setSigma(Number(e.target.value))} />
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Image A (mask = 1)</h3>
          <PixelCanvas image={imageA?.imageData ?? null} size={INPUT_DISPLAY} pixelated={false} label="Image A" />
        </div>
        <div className="demo-panel">
          <h3>Image B (mask = 0)</h3>
          <PixelCanvas image={imageBShown} size={INPUT_DISPLAY} pixelated={false} label="Image B, resized to the size of A" placeholderAspect={aspect} />
          {imageA && imageB && (imageB.width !== imageA.width || imageB.height !== imageA.height) && (
            <p className="caption">Resized {imageB.width}×{imageB.height} → {imageA.width}×{imageA.height}</p>
          )}
        </div>
        <div className="demo-panel">
          <h3>Mask</h3>
          <PixelCanvas image={maskShown} size={INPUT_DISPLAY} pixelated={false} label="Blend mask, white takes image A" placeholderAspect={aspect}
            placeholder={customMask ? 'Loading mask…' : undefined} />
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Hard cut</h3>
          <PixelCanvas image={hard} size={DISPLAY} pixelated={false} label="Hard cut: A where the mask is white, B elsewhere" placeholderAspect={aspect} />
          <p className="caption">Visible seam</p>
        </div>
        <div className="demo-panel">
          <h3>Feathered (<span className="pyr-keep-case">σ</span> = {sigma})</h3>
          <PixelCanvas image={feathered} size={DISPLAY} pixelated={false} label="Feathered blend with a Gaussian-blurred mask" placeholderAspect={aspect} />
          <p className="caption">One transition width for all frequencies</p>
        </div>
        <div className="demo-panel">
          <h3>Multiband ({n} level{n > 1 ? 's' : ''})</h3>
          <PixelCanvas image={multiband?.image ?? null} size={DISPLAY} pixelated={false} label="Multiband Laplacian pyramid blend" placeholderAspect={aspect} />
          <p className="caption">Each band blended over its own width</p>
        </div>
      </div>

      <div className="demo-panel">
        <h3>Inside the multiband blend</h3>
        <div className="pyr-levels pyr-bands">
          {maskLevels.slice(0, n).map((m, i) => (
            <figure key={i} className="pyr-level">
              {/* Levels smaller than the display are drawn pixelated, so their coarse grid stays visible. */}
              <PixelCanvas image={m} size={LEVEL_DISPLAY} pixelated={Math.max(m.width, m.height) < LEVEL_DISPLAY} label={`Mask, Gaussian level ${i}`} />
              {bandsShown[i] && (
                <PixelCanvas image={bandsShown[i]} size={LEVEL_DISPLAY} pixelated={Math.max(m.width, m.height) < LEVEL_DISPLAY}
                  label={i === n - 1 ? `Blended low-pass residual, level ${i}` : `Blended Laplacian band ${i}`} />
              )}
              <figcaption>level {i}<br />{m.width}×{m.height}</figcaption>
            </figure>
          ))}
        </div>
        <p className="caption">Top: the mask's Gaussian pyramid Mᵢ, all shown at the same size. Bottom: Mᵢ·LAᵢ + (1 − Mᵢ)·LBᵢ
        (bands as 128 + 2·L; the last is the blended low-pass residual). Summing these back up gives the multiband result.
        Coarse levels look shifted slightly right and down: reduce keeps pixels 0, 2, 4, …, so each coarse pixel samples
        the top-left corner of the block it is drawn over.</p>
      </div>

      <div className="demo-explanation">
        <h4>Why multiband blending hides the seam</h4>
        <p>A <strong>hard cut</strong> jumps from A to B in one pixel: any difference in brightness or colour becomes a sharp edge.
        <strong> Feathering</strong> blurs the mask, but must pick one width: narrow (small σ) still shows a step in the smooth
        sky colours; wide (large σ) blends fine textures over a broad strip where both images show at once (ghosting).</p>
        <p><strong>Multiband blending</strong> (Burt &amp; Adelson, 1983) picks the width per frequency band. Both images are split
        into Laplacian pyramids, the mask into a Gaussian pyramid. Level i is blended with mask level i, whose transition spans
        about 2<sup>i</sup> original pixels — so <strong>low frequencies</strong> (overall colour, lighting) mix over a wide area and
        the seam's colour step fades out, while <strong>high frequencies</strong> (edges, texture) switch over a narrow strip and
        stay crisp without ghosts. Collapsing the blended pyramid gives the result. With 1 level and a binary mask it is the
        hard cut (with a soft, uploaded mask, a plain alpha blend); more levels widen the blend of the coarsest band.</p>
        {imageA && levelLimit > 1 && (
          <p>Why does the slider stop at {levelLimit}? Each extra level doubles the width of the coarsest blend; here the
          coarsest level is {coarsest}, still large enough to show the mask's shape. A 2×2 or 1×1 residual would be little more
          than each image's mean colour, blended with a mask level close to 0.5 everywhere: A's colours would spread into B's
          side and B's into A's. And since the coarse grid starts at the top-left pixel, that leak would be lopsided.</p>
        )}
        {imageA && levelLimit === 1 && (
          <p>Why is there only 1 level? Image A is {imageA.width}×{imageA.height}: one halving would already make its shorter
          side smaller than 16 px, too coarse to show the mask's shape, so the blend stays a hard cut. Upload a larger image A to
          get more levels.</p>
        )}
      </div>
    </>
  );
}
