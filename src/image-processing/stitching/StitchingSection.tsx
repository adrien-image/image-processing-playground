import { useMemo, useState } from 'react';
import homographyCode from './homography.ts?raw';
import pythonCode from './stitching.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { toImageData } from '../../lib/canvas';
import { makeCanvas } from '../../lib/demoImages';
import { seededRandom } from '../../lib/random';
import { rgbaToGrayscale } from '../processors/utils';
import { detectHarris } from '../matching/harris';
import {
  applyHomography, cameraHomography, fitHomography, matchPoints, ransacHomography, renderView, stitch, vignette, type Blend, type Homography, type Point, type RgbImage,
} from './homography';
import './stitching.css';

const SCENE_MAX = 400;
const VIEW_FRACTION = 0.62;

/**
 * A wide street: sky, hills, buildings and signs. Windows vary in size, brightness and
 * spacing, so their corners differ from each other (identical windows would all match
 * each other equally well and fail the ratio test).
 */
function street(): HTMLCanvasElement {
  const W = 400, H = 220;
  // Drawn at twice the size and shrunk on load: antialiased edges survive resampling better.
  return makeCanvas(2 * W, 2 * H, ctx => {
    ctx.scale(2, 2);
    const random = seededRandom(11);
    const r = (lo: number, hi: number) => lo + random() * (hi - lo);
    const sky = ctx.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0, '#3b5b8c');
    sky.addColorStop(1, '#c9d6e8');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#eef2f7';
    for (let i = 0; i < 6; i++) {
      const cx = r(10, W - 10), cy = r(12, 50);
      for (let k = 0; k < 4; k++) { ctx.beginPath(); ctx.arc(cx + r(-14, 14), cy + r(-4, 4), r(5, 11), 0, Math.PI * 2); ctx.fill(); }
    }
    ctx.fillStyle = '#5d6b5a';
    ctx.beginPath();
    ctx.moveTo(0, 120);
    for (let x = 0; x <= W; x += 10) ctx.lineTo(x, 105 - 25 * Math.sin(x / 45) - 10 * Math.sin(x / 13) + r(-4, 4));
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.fill();
    for (let x = 2; x < W - 10;) {
      const bw = Math.floor(r(24, 56)), bh = Math.floor(r(55, 150));
      ctx.fillStyle = `hsl(${Math.floor(r(0, 360))} ${Math.floor(r(10, 45))}% ${Math.floor(r(18, 48))}%)`;
      ctx.fillRect(x, H - 20 - bh, bw, bh);
      const ww = Math.floor(r(4, 10)), wh = Math.floor(r(5, 11)), gx = ww + Math.floor(r(4, 9)), gy = wh + Math.floor(r(4, 9));
      for (let wy = H - 14 - bh; wy < H - 30; wy += gy)
        for (let wx = x + 3; wx < x + bw - ww - 2; wx += gx) {
          if (random() < 0.3) continue;
          ctx.fillStyle = `hsl(${Math.floor(r(30, 60))} ${Math.floor(r(20, 90))}% ${Math.floor(r(45, 92))}%)`;
          ctx.fillRect(wx, wy, ww, wh);
        }
      if (random() < 0.5) {                         // a sign or a door
        ctx.fillStyle = `hsl(${Math.floor(r(0, 360))} 70% 55%)`;
        ctx.fillRect(x + r(2, bw / 2), H - 20 - r(8, 40), r(8, 18), r(5, 10));
      }
      x += bw + Math.floor(r(1, 10));
    }
    ctx.fillStyle = '#2b2b30';
    ctx.fillRect(0, H - 20, W, 20);
    ctx.fillStyle = '#e8e8e8';
    for (let x = 6; x < W; x += Math.floor(r(22, 34))) ctx.fillRect(x, H - 11, Math.floor(r(8, 16)), 3);
  });
}

function crop(img: RgbImage, width: number): RgbImage {
  const data = new Uint8ClampedArray(width * img.height * 4);
  for (let y = 0; y < img.height; y++) data.set(img.data.subarray(y * img.width * 4, (y * img.width + width) * 4), y * width * 4);
  return { data, width, height: img.height };
}

/** A and B side by side with a small gap, as one image (for the match lines). */
function sideBySide(a: RgbImage, b: RgbImage, gap: number) {
  const width = a.width + gap + b.width, height = Math.max(a.height, b.height);
  const data = new Uint8ClampedArray(width * height * 4);
  for (const [img, ox] of [[a, 0], [b, a.width + gap]] as const)
    for (let y = 0; y < img.height; y++) data.set(img.data.subarray(y * img.width * 4, (y + 1) * img.width * 4), (y * width + ox) * 4);
  return toImageData(data, width, height);
}

const meanCornerError = (H: Homography, truth: Homography, w: number, h: number) =>
  [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]].reduce((s, [x, y]) => {
    const a = applyHomography(H, { x, y }), b = applyHomography(truth, { x, y });
    return s + Math.hypot(a.x - b.x, a.y - b.y);
  }, 0) / 4;

export function StitchingSection() {
  const [source, setSource] = useState<ImageSource>(street);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [angle, setAngle] = useState(10);
  const [corners, setCorners] = useState(250);
  const [threshold, setThreshold] = useState(2);
  const [useRansac, setUseRansac] = useState(true);
  const [blend, setBlend] = useState<Blend>('feather');
  const [vignetting, setVignetting] = useState(0.25);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, SCENE_MAX);

  const views = useMemo(() => {
    if (!image) return null;
    const scene: RgbImage = { data: image.imageData.data, width: image.width, height: image.height };
    const vw = Math.round(image.width * VIEW_FRACTION);
    const a = vignette(crop(scene, vw), vignetting);
    // Camera B is turned by `angle` (and shifted so its centre always lands 45% of a view
    // further right, keeping the overlap): B(x) = scene(G·x), so G maps B's pixels into A's.
    const focal = vw * 1.1;
    const shift = Math.round(vw * 0.45) - focal * Math.tan((angle * Math.PI) / 180);
    const truth = cameraHomography(angle, shift, focal, vw / 2, image.height / 2);
    const b = vignette(renderView(scene, truth, vw, image.height), vignetting);
    return { a, b, truth, grayA: rgbaToGrayscale(a.data, vw, a.height), grayB: rgbaToGrayscale(b.data, vw, b.height) };
  }, [image, angle, vignetting]);

  const matches = useMemo(() => {
    if (!views) return null;
    const { a, grayA, grayB } = views;
    const kpA = detectHarris(grayA, a.width, a.height, corners);
    const kpB = detectHarris(grayB, a.width, a.height, corners);
    const m = matchPoints(grayA, kpA, grayB, kpB, a.width, a.height);
    return { inA: m.a, inB: m.b };
  }, [views, corners]);

  const fit = useMemo(() => {
    if (!matches || !views) return null;
    if (useRansac) return ransacHomography(matches.inB, matches.inA, { threshold });
    const H = fitHomography(matches.inB, matches.inA);
    return H && { H, inliers: matches.inA.map(() => true), inlierCount: matches.inA.length };
  }, [matches, views, useRansac, threshold]);

  const panorama = useMemo(() => {
    if (!views || !fit) return null;
    const p = stitch(views.a, views.b, fit.H, blend);
    return toImageData(p.data, p.width, p.height);
  }, [views, fit, blend]);

  const pair = useMemo(() => (views ? sideBySide(views.a, views.b, 8) : null), [views]);
  const error = uploadError ?? loadError;
  const cornerError = fit && views ? meanCornerError(fit.H, views.truth, views.a.width, views.a.height) : null;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <label className="dl-slider">Camera turn <strong>{angle}°</strong>
          <input type="range" min={0} max={25} value={angle} onChange={e => setAngle(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Corners per image <strong>{corners}</strong>
          <input type="range" min={20} max={300} step={10} value={corners} onChange={e => setCorners(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Lens vignetting <strong>{Math.round(vignetting * 100)}%</strong>
          <input type="range" min={0} max={0.6} step={0.05} value={vignetting} onChange={e => setVignetting(Number(e.target.value))} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={useRansac} onChange={e => setUseRansac(e.target.checked)} /> RANSAC</label>
        {useRansac && (
          <label className="dl-slider">Inlier threshold <strong>{threshold} px</strong>
            <input type="range" min={0.5} max={10} step={0.5} value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
          </label>
        )}
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setUploadError(null); }} onError={setUploadError} />
        <span className="ml-auto">
          <ShowCodeButton code={homographyCode} title="stitching/homography.ts" python={{ code: pythonCode, title: 'stitching/stitching.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {views && pair && matches && (
        <div className="demo-panel">
          <h3>Two photos and their matches</h3>
          <PixelCanvas image={pair} size={720} pixelated={false} label="Left and right photos with lines joining matched corners">
            <svg className="stitch-lines" viewBox={`0 0 ${pair.width} ${pair.height}`} preserveAspectRatio="none" aria-hidden="true">
              {matches.inA.map((p: Point, i) => {
                const q = matches.inB[i], ok = fit?.inliers[i];
                return <line key={i} x1={p.x + 0.5} y1={p.y + 0.5} x2={q.x + views.a.width + 8.5} y2={q.y + 0.5} className={ok ? 'stitch-inlier' : 'stitch-outlier'} />;
              })}
            </svg>
          </PixelCanvas>
          <p className="caption"><span className="stitch-key stitch-key-in" /> agrees with the homography · <span className="stitch-key stitch-key-out" /> outlier</p>
        </div>
      )}

      <div className="dl-stat-row">
        <span>Matches <strong>{matches?.inA.length ?? 0}</strong></span>
        <span>Inliers <strong>{fit?.inlierCount ?? 0}</strong></span>
        <span>Corner error vs. true H <strong>{cornerError === null ? '—' : `${cornerError.toFixed(1)} px`}</strong></span>
      </div>

      <div className="demo-panel">
        <div className="stitch-panorama-head">
          <h3>Panorama</h3>
          <span className="demo-label">Blend:</span>
          {(['overwrite', 'average', 'feather'] as const).map(b => (
            <button key={b} type="button" aria-pressed={blend === b} className={`btn btn-sm${blend === b ? ' btn-primary' : ''}`} onClick={() => setBlend(b)}>
              {b === 'overwrite' ? 'Overwrite' : b === 'average' ? 'Average' : 'Feather'}
            </button>
          ))}
        </div>
        {panorama
          ? <PixelCanvas image={panorama} size={720} pixelated={false} label="The two photos warped onto one canvas" />
          : <p className="dl-status">Not enough matches to fit a homography.</p>}
      </div>

      <div className="demo-explanation">
        <h4>Stitching a panorama</h4>
        <p>Turn a camera on a tripod and every photo is related to the next by a <strong>homography</strong>: a 3×3 matrix acting on
        homogeneous pixel coordinates. Straight lines stay straight, but parallel lines can converge, which is why the warped photo looks
        like a trapezoid. (The same holds for any camera motion when the scene is flat.)</p>
        <p className="formula"><code>(x′, y′, w′) = H · (x, y, 1)   →   (x′/w′, y′/w′)</code></p>
        <p>H has 8 degrees of freedom, so 4 matched points fix it. The pipeline: detect Harris corners in both photos, describe and match
        them (as in the Feature matching lesson), then solve for H. Some matches are always wrong, and least squares on all of them is
        pulled far off: untick <strong>RANSAC</strong> to see it. RANSAC instead fits H to random sets of 4 matches, keeps the one most
        matches agree with (within the threshold), and refits on those inliers.</p>
        <p>Finally each panorama pixel looks up where it comes from in each photo (inverse warping, bilinear sampling). Where both
        cover it, the photos rarely agree exactly: here the lens darkens each photo's corners (vignetting), so the edge
        of one photo meets the brighter centre of the other. <strong>Overwrite</strong> leaves a hard seam, <strong>average</strong> shows a step where one photo ends, and
        <strong> feather</strong> weights each photo by the distance to its own border so the seam fades away. Real stitchers also
        correct exposure differences and choose seams that avoid moving objects.</p>
      </div>
    </div>
  );
}
