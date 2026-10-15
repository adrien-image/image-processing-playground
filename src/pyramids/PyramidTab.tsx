import { useMemo, useState } from 'react';
import { PixelCanvas } from '../components/PixelCanvas';
import { ImageUploader } from '../components/ImageUploader';
import { useImageData, type ImageSource } from '../hooks/useImageData';
import { maxAbsDifference, splitRGB } from './plane';
import { laplacianLevels, maxLevels, reconstruct } from './pyramid';
import { byLevel, DEMO_SIZE, planesToImageData, pyramidsPerChannel, sunsetCanvas, zonePlateCanvas } from './view';
import { CodeMenu } from './CodeMenu';

const DISPLAY = 256;
const EQUAL_SIZE = 128;
/** "True size" thumbnails halve per level, but are never drawn smaller than this. */
const MIN_THUMB = 4;
const MAX_LEVELS = 8;
const GAINS = [1, 2, 4] as const;

type Demo = 'scene' | 'zone';
const DEMOS: { key: Demo; label: string; make: () => HTMLCanvasElement }[] = [
  { key: 'scene', label: 'Colour scene', make: sunsetCanvas },
  { key: 'zone', label: 'Zone plate', make: zonePlateCanvas },
];

/** 0 → "0 (exact)", tiny values in scientific notation. */
const formatError = (e: number) => (e === 0 ? '0 (exact)' : e.toExponential(1));

interface LevelProps {
  image: ImageData;
  name: string;
  index: number;
  size: number;
  /** How much larger than the halving staircase this level is drawn (1 = to scale). */
  zoom: number;
  label: string;
}

/** One pyramid level: image + "Gᵢ · w×h" caption (+ "×2" when drawn larger than to scale). */
function Level({ image, name, index, size, zoom, label }: LevelProps) {
  return (
    <figure className="pyr-level">
      {/* Pixelated only when enlarged: shrinking with nearest-neighbour would add moiré of its own. */}
      <PixelCanvas image={image} size={size} label={label} pixelated={size >= Math.max(image.width, image.height)}
        className={size < 16 ? 'pyr-tiny' : undefined} />
      <figcaption>
        {name}<sub>{index}</sub> · {image.width}×{image.height}
        {zoom > 1 && <span className="pyr-zoom" title={`Drawn ${zoom}× larger than to scale`}> ×{zoom}</span>}
      </figcaption>
    </figure>
  );
}

export function PyramidTab() {
  const [demo, setDemo] = useState<Demo>('scene');
  const [demoSource, setDemoSource] = useState<ImageSource>(sunsetCanvas);
  const [upload, setUpload] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [levels, setLevels] = useState(5);
  const [equalSize, setEqualSize] = useState(false);
  const [gain, setGain] = useState<number>(2);

  const { image, error: loadError } = useImageData(upload ?? demoSource, DEMO_SIZE);

  const planes = useMemo(() => (image ? splitRGB(image.imageData.data, image.width, image.height) : null), [image]);
  const levelLimit = image ? Math.min(MAX_LEVELS, maxLevels(image.width, image.height)) : MAX_LEVELS;
  const n = Math.min(levels, levelLimit);

  // Stage 1 (image only): the deepest pyramids, per channel. Fewer levels are a slice of
  // these — only the residual and the reconstruction depend on the slider.
  const deep = useMemo(() => (planes ? pyramidsPerChannel(planes, levelLimit) : null), [planes, levelLimit]);

  // Stage 2 (levels): rebuild the image from n levels and measure the error.
  const rebuilt = useMemo(() => {
    if (!deep || !planes) return null;
    const channels = deep.laplacian.map((lap, c) => reconstruct(laplacianLevels(deep.gaussian[c], lap, n)));
    return { image: planesToImageData(channels), error: Math.max(...channels.map((r, c) => maxAbsDifference(r, planes[c]))) };
  }, [deep, planes, n]);

  // Display images, made once per image; the contrast toggle only redraws the bands.
  const gaussianImages = useMemo(() => (deep ? byLevel(deep.gaussian).map(ps => planesToImageData(ps)) : []), [deep]);
  const bandImages = useMemo(() => (deep ? byLevel(deep.laplacian).map(ps => planesToImageData(ps, 128, gain)) : []), [deep, gain]);
  // n levels = bands L₀ … L_{n−2}, then the residual G_{n−1}.
  const laplacianShown = [...bandImages.slice(0, n - 1), ...gaussianImages.slice(n - 1, n)];

  // "True size": the display halves with every level, like the images (256, 128, …).
  const thumb = (i: number) => (equalSize ? EQUAL_SIZE : Math.max(MIN_THUMB, Math.round(DISPLAY / 2 ** i)));
  const zoom = (i: number) => (equalSize ? 1 : Math.round(thumb(i) / (DISPLAY / 2 ** i)));
  const error = uploadError ?? loadError;
  const exact = rebuilt !== null && rebuilt.error < 1e-6;

  return (
    <>
      <div className="demo-controls">
        <span className="demo-label">Image:</span>
        {DEMOS.map(d => {
          const active = !upload && demo === d.key;
          return (
            <button key={d.key} type="button" aria-pressed={active} className={`btn btn-sm${active ? ' btn-primary' : ''}`}
              onClick={() => {
                if (d.key !== demo) { setDemo(d.key); setDemoSource(d.make()); } // same demo: keep its canvas (and pyramids)
                setUpload(null);
                setUploadError(null);
              }}>
              {d.label}
            </button>
          );
        })}
        <ImageUploader label="📁 Upload" onImage={src => { setUpload(src); setUploadError(null); }} onError={setUploadError} />
        <span className="demo-label">Thumbnails:</span>
        <button type="button" aria-pressed={!equalSize} className={`btn btn-sm${!equalSize ? ' btn-primary' : ''}`} onClick={() => setEqualSize(false)}>
          True size
        </button>
        <button type="button" aria-pressed={equalSize} className={`btn btn-sm${equalSize ? ' btn-primary' : ''}`} onClick={() => setEqualSize(true)}>
          Same size
        </button>
        <CodeMenu initial="pyramid.ts" />
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="pyr-levels">Levels: <strong>{image ? n : '–'}</strong> (smallest {gaussianImages[n - 1]?.width ?? '–'}×{gaussianImages[n - 1]?.height ?? '–'})</label>
          <input id="pyr-levels" type="range" className="demo-slider" min={Math.min(2, levelLimit)} max={levelLimit} value={n}
            disabled={!image || levelLimit < 2} onChange={e => setLevels(Number(e.target.value))} />
        </div>
      </div>

      <div className="demo-panel">
        <h3>Gaussian pyramid</h3>
        <p className="caption pyr-formula">Gᵢ₊₁ = reduce(Gᵢ): blur with [1 4 6 4 1]/16, keep every 2nd pixel</p>
        <div className="pyr-levels">
          {gaussianImages.slice(0, n).map((img, i) => (
            <Level key={i} image={img} name="G" index={i} size={thumb(i)} zoom={zoom(i)} label={`Gaussian level ${i}, ${img.width} by ${img.height}`} />
          ))}
        </div>
      </div>

      <div className="demo-panel">
        <h3>Laplacian pyramid</h3>
        <p className="caption pyr-formula">Lᵢ = Gᵢ − expand(Gᵢ₊₁): the detail lost by one halving</p>
        <div className="segmented" role="group" aria-label="Laplacian band contrast">
          <span className="demo-label">Band contrast:</span>
          {GAINS.map(g => (
            <button key={g} type="button" aria-pressed={gain === g} className={`btn btn-sm${gain === g ? ' btn-primary' : ''}`} onClick={() => setGain(g)}>
              ×{g}
            </button>
          ))}
        </div>
        <div className="pyr-levels">
          {laplacianShown.map((img, i) => (
            <Level key={i} image={img} name={i === n - 1 ? 'G' : 'L'} index={i} size={thumb(i)} zoom={zoom(i)}
              label={i === n - 1 ? `Low-pass residual, Gaussian level ${i}` : `Laplacian band ${i}, zero shown as mid-gray`} />
          ))}
        </div>
        <p className="caption">Bands: 128 + {gain}·Lᵢ (mid-gray = 0). The last level is the smallest Gaussian, kept as it is.</p>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Original image" />
        </div>
        <div className="demo-panel">
          <h3>Reconstructed</h3>
          <PixelCanvas image={rebuilt?.image ?? null} size={DISPLAY} label="Image rebuilt by expanding and adding the Laplacian bands" />
          <p className="caption">
            Expand and add the bands back. Max |original − reconstructed| ={' '}
            <strong className={exact ? 'text-success' : 'text-error'}>{rebuilt ? formatError(rebuilt.error) : '–'}</strong> (0–255 scale)
          </p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Gaussian and Laplacian pyramids</h4>
        <p><strong>reduce</strong> blurs with the binomial filter [1 4 6 4 1]/16 (rows, then columns) and keeps every second pixel, so each
        <strong> Gaussian</strong> level is a half-size, nearly alias-free copy of the one above: ⌈w/2⌉ × ⌈h/2⌉. The 5-tap filter is a
        cheap compromise: it still passes 25 % at the new Nyquist limit, so a trace of aliasing can remain (on the zone plate, look
        at G<sub>1</sub>'s edges). <strong>expand</strong> goes back up:
        put the small image on the even pixels, zeros in between, and blur with 4 × the same filter (the 4 makes up for the zeros).</p>
        <p>A <strong>Laplacian</strong> level is the difference between a Gaussian level and the expanded next one: the detail that
        halving threw away, i.e. one band of frequencies (an octave). On the zone plate each band picks out one ring of frequencies;
        on the scene, L₀ holds the sharpest edges and coarser bands hold broader shapes. Most band pixels are near 0
        (mid-gray), which is why the pyramid was invented for compression.</p>
        <p>The pyramid loses nothing: starting from the smallest level, expand and add each band back. Since every band was
        defined against exactly that expand, the image comes back up to floating-point rounding — often bit for bit, as the
        weights (1/16, 4/16, 6/16, 1/8, 1/2) are exact binary fractions. Each colour channel is processed separately.</p>
      </div>
    </>
  );
}
