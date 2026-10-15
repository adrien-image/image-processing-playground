import { useMemo, useState, type MouseEvent } from 'react';
import jpegCode from './jpeg.ts?raw';
import pythonCode from './jpeg.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { scene } from '../../lib/demoImages';
import { divergingColor } from '../../lib/colormap';
import { grayToImageData, rgbImage } from '../../lib/image';
import { hotColor } from '../hough/hough';
import { psnr } from '../denoising/denoise';
import { dct2, estimateBits, idct2, jpeg, quantTable, ZIGZAG } from './jpeg';
import './jpeg.css';

const DISPLAY = 300;

/** The 64 DCT basis patterns, one 8×8 tile each with a 1-pixel gap (frequency increases right and down). */
const BASIS = (() => {
  const SIZE = 8 * 9 - 1;
  const values = new Float64Array(SIZE * SIZE).fill(NaN);
  for (let k = 0; k < 64; k++) {
    const unit = new Float64Array(64);
    unit[k] = 1;
    const pattern = idct2(unit);
    const kx = k % 8, ky = Math.floor(k / 8);
    for (let p = 0; p < 64; p++) values[(ky * 9 + Math.floor(p / 8)) * SIZE + kx * 9 + (p % 8)] = pattern[p];
  }
  return rgbImage(SIZE, SIZE, i => {
    if (Number.isNaN(values[i])) return [20, 24, 36];
    const v = 128 + values[i] * 4 * 127;
    return [v, v, v];
  });
})();

export function JpegSection() {
  const [source, setSource] = useState<ImageSource>(() => scene(256));
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [quality, setQuality] = useState(30);
  const [blockAt, setBlockAt] = useState({ bx: 19, by: 29 }); // on the "Aa": plenty of detail
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, 256);

  const result = useMemo(() => {
    if (!image) return null;
    const { gray, width, height } = image;
    const r = jpeg(gray, width, height, quality);
    const bits = estimateBits(r);
    const diff = r.output.map((v, i) => Math.abs(v - gray[i]));
    return {
      r, bits,
      psnr: psnr(gray, r.output),
      decoded: grayToImageData(r.output, width, height),
      diff: rgbImage(width, height, i => hotColor(Math.min(1, diff[i] / 32))),
    };
  }, [image, quality]);

  const table = useMemo(() => quantTable(quality), [quality]);
  const inspector = useMemo(() => {
    if (!image || !result) return null;
    const { gray, width, height } = image;
    const bx = Math.min(blockAt.bx, result.r.blocksX - 1), by = Math.min(blockAt.by, result.r.blocksY - 1);
    const pixels = new Float64Array(64);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      pixels[y * 8 + x] = gray[Math.min(height - 1, by * 8 + y) * width + Math.min(width - 1, bx * 8 + x)] - 128;
    }
    const coeffs = dct2(pixels);
    const q = result.r.quantised.subarray((by * result.r.blocksX + bx) * 64, (by * result.r.blocksX + bx + 1) * 64);
    const decoded = idct2(Array.from(q, (v, i) => v * table[i]));
    return { bx, by, pixels, coeffs, q: Array.from(q), decoded };
  }, [image, result, blockAt, table]);

  const pick = (e: MouseEvent<HTMLDivElement>) => {
    if (!image) return;
    const rect = e.currentTarget.getBoundingClientRect();
    setBlockAt({
      bx: Math.floor(((e.clientX - rect.left) / rect.width) * image.width / 8),
      by: Math.floor(((e.clientY - rect.top) / rect.height) * image.height / 8),
    });
  };

  const error = uploadError ?? loadError;
  const raw = image ? image.width * image.height : 1;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <label className="demo-slider-label" htmlFor="jpeg-q">Quality <strong>{quality}</strong></label>
        <input id="jpeg-q" type="range" min={1} max={100} value={quality} onChange={e => setQuality(Number(e.target.value))} className="demo-slider" />
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setUploadError(null); }} onError={setUploadError} />
        <button type="button" className="btn btn-sm" onClick={() => setSource(scene(256))}>Demo image</button>
        <span className="ml-auto">
          <ShowCodeButton code={jpegCode} title="jpeg/jpeg.ts" python={{ code: pythonCode, title: 'jpeg/jpeg.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {result && image && (
        <div className="demo-images">
          <div className="demo-panel">
            <h3>Original (8 bits per pixel)</h3>
            <PixelCanvas image={image.imageData} size={DISPLAY} label="Original image" />
          </div>
          <div className="demo-panel">
            <h3>JPEG, quality {quality}</h3>
            <PixelCanvas image={result.decoded} size={DISPLAY} label={`Image after JPEG compression at quality ${quality}`}>
              <div className="jpeg-picker" onClick={pick} role="presentation">
                {inspector && (
                  <div className="jpeg-block" style={{
                    left: `${(inspector.bx * 8 / image.width) * 100}%`, top: `${(inspector.by * 8 / image.height) * 100}%`,
                    width: `${(8 / image.width) * 100}%`, height: `${(8 / image.height) * 100}%`,
                  }} />
                )}
              </div>
            </PixelCanvas>
            <p className="caption">Click an 8×8 block to inspect it below.</p>
          </div>
          <div className="demo-panel">
            <h3>Error (|difference|, black = none)</h3>
            <PixelCanvas image={result.diff} size={DISPLAY} label="Where the compressed image differs from the original" />
          </div>
        </div>
      )}

      {result && (
        <div className="jpeg-stats">
          <div><strong>{result.psnr.toFixed(1)} dB</strong><span>PSNR</span></div>
          <div><strong>{(result.bits / 8 / 1024).toFixed(1)} KB</strong><span>estimated size</span></div>
          <div><strong>{(raw * 8 / result.bits).toFixed(1)}×</strong><span>smaller than raw</span></div>
          <div><strong>{(result.bits / raw).toFixed(2)}</strong><span>bits per pixel</span></div>
          <div><strong>{((result.r.nonZero / (result.r.blocksX * result.r.blocksY * 64)) * 100).toFixed(1)}%</strong><span>coefficients kept</span></div>
        </div>
      )}

      {inspector && (
        <div className="demo-panel jpeg-inspector">
          <h3>Block ({inspector.bx}, {inspector.by}), step by step</h3>
          <div className="jpeg-steps">
            <BlockGrid title="1 · pixels − 128" values={inspector.pixels} scale={128} />
            <span className="jpeg-arrow" aria-hidden="true">DCT →</span>
            <BlockGrid title="2 · DCT coefficients" values={inspector.coeffs} scale={200} />
            <span className="jpeg-arrow" aria-hidden="true">÷ →</span>
            <BlockGrid title={`3 · quantisation steps (q ${quality})`} values={table} scale={0} />
            <span className="jpeg-arrow" aria-hidden="true">round →</span>
            <BlockGrid title="4 · quantised (what is stored)" values={inspector.q} scale={8} zeroDim />
            <span className="jpeg-arrow" aria-hidden="true">× steps, IDCT →</span>
            <BlockGrid title="5 · decoded − 128" values={inspector.decoded} scale={128} />
          </div>
          <p className="caption">
            Stored in zigzag order: {ZIGZAG.map(i => inspector.q[i]).join(' ').replace(/( 0)+$/, '')} · EOB
          </p>
        </div>
      )}

      <div className="demo-panel jpeg-basis">
        <h3>The 64 DCT patterns</h3>
        <PixelCanvas image={BASIS} size={288} label="The 64 cosine patterns of the 8×8 DCT, frequency increasing right and down" />
        <p className="caption">Every 8×8 block is a weighted sum of these. Coefficient (row, column) is the weight of the pattern at that position.</p>
      </div>

      <div className="demo-explanation">
        <h4>Where JPEG throws information away</h4>
        <p>The <strong>DCT</strong> (discrete cosine transform) rewrites each 8×8 block as weights of the 64 patterns above. That step
        loses nothing. But in natural images most of the energy lands in the few top-left (low-frequency) weights. The loss happens in
        <strong> quantisation</strong>: each weight is divided by a step and rounded. The steps grow towards high frequencies, which the eye
        barely sees, so most of those weights become 0, and long runs of zeros cost almost nothing to store.</p>
        <p>Lower the quality to watch the classic artefacts appear: <strong>blocking</strong> (each 8×8 block decoded on its own, so their edges
        no longer line up) and <strong>ringing</strong> around sharp edges, where the missing high frequencies can't rebuild the step. Smooth
        areas compress well; text and fine stripes don't. Real JPEG also converts colour to brightness + two colour channels and stores the colour
        ones at half resolution, because the eye is much less sensitive to colour detail. This lesson keeps to brightness.</p>
      </div>
    </div>
  );
}

function BlockGrid({ title, values, scale, zeroDim = false }: { title: string; values: ArrayLike<number>; scale: number; zeroDim?: boolean }) {
  return (
    <figure className="jpeg-grid-wrap">
      <figcaption>{title}</figcaption>
      <div className="jpeg-grid">
        {Array.from(values, (v, i) => {
          const [r, g, b] = scale > 0 ? divergingColor(v / scale) : [40, 46, 80];
          return (
            <span key={i} className={`jpeg-cell${zeroDim && Math.round(v) === 0 ? ' jpeg-cell-zero' : ''}`}
              style={{ background: `rgb(${r * 0.7}, ${g * 0.7}, ${b * 0.7})` }}>
              {Math.round(v)}
            </span>
          );
        })}
      </div>
    </figure>
  );
}
