import { useMemo, useState } from 'react';
import calibrationCode from './calibration.ts?raw';
import pythonCode from './calibration.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { toImageData } from '../../lib/canvas';
import { gaussianRandom, seededRandom } from '../../lib/random';
import { applyHomography, type Point } from '../stitching/homography';
import {
  boardCorners, boardPose, calibrate, distortImage, projectCorners, radialShift, renderCheckerboard, undistortImage,
  type Calibration, type Camera, type Distortion,
} from './calibration';
import './calibration.css';

const W = 200, H = 150;
const DISPLAY = 340;
const COLS = 9, ROWS = 6;
const CAM: Camera = { f: 200, cx: W / 2, cy: H / 2 };
const BOARD = boardCorners(COLS, ROWS);

const PRESETS: { label: string; k: Distortion }[] = [
  { label: 'Barrel (wide angle)', k: { k1: -0.35, k2: 0.1 } },
  { label: 'Pincushion (telephoto)', k: { k1: 0.3, k2: 0 } },
  { label: 'Moustache', k: { k1: -0.4, k2: 0.6 } },
  { label: 'None', k: { k1: 0, k2: 0 } },
];

const fmt = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);

/** Straight lines through each row and column of corners (they should be straight once undistorted). */
function GridLines({ points, className }: { points: Point[]; className: string }) {
  const rows = Array.from({ length: ROWS }, (_, j) => points.slice(j * COLS, (j + 1) * COLS));
  const cols = Array.from({ length: COLS }, (_, i) => rows.map(r => r[i]));
  return (
    <>
      {[...rows, ...cols].map((line, i) => (
        <polyline key={i} className={className} points={line.map(p => `${p.x},${p.y}`).join(' ')} />
      ))}
    </>
  );
}

export function CalibrationSection() {
  const [k, setK] = useState<Distortion>(PRESETS[0].k);
  const [rotate, setRotate] = useState(10);
  const [tilt, setTilt] = useState(0.35);
  const [scale, setScale] = useState(19);
  const [noise, setNoise] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const [result, setResult] = useState<{ calibration: Calibration; key: string } | null>(null);
  const [showCode, setShowCode] = useState(false);

  const pose = useMemo(() => boardPose(COLS, ROWS, CAM.cx, CAM.cy, scale, rotate, tilt), [scale, rotate, tilt]);
  const photo = useMemo(() => distortImage(renderCheckerboard(W, H, pose, COLS, ROWS), W, H, CAM, k), [pose, k]);
  const photoImage = useMemo(() => toImageData(photo, W, H), [photo]);

  // "Detected" corners: where the lens really puts them, plus detection noise.
  const observed = useMemo(() => {
    const random = seededRandom(seed);
    return projectCorners(BOARD, pose, CAM, k).map(p => ({ x: p.x + noise * gaussianRandom(random), y: p.y + noise * gaussianRandom(random) }));
  }, [pose, k, noise, seed]);

  const key = JSON.stringify([k, rotate, tilt, scale, noise, seed]);
  const calibration = result?.key === key ? result.calibration : null;

  const corrected = useMemo(() => (calibration ? toImageData(undistortImage(photo, W, H, CAM, calibration.k), W, H) : null), [photo, calibration]);
  const straight = useMemo(() => (calibration ? BOARD.map(p => applyHomography(calibration.H, p)) : null), [calibration]);

  const radii = Array.from({ length: 26 }, (_, i) => (i / 25) * Math.hypot(CAM.cx, CAM.cy));
  const shifts = (d: Distortion) => radii.map(r => radialShift(CAM, d, r));

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Lens:</span>
        {PRESETS.map(p => (
          <button key={p.label} type="button" aria-pressed={k === p.k} className={`btn btn-sm${k === p.k ? ' btn-primary' : ''}`}
            onClick={() => setK(p.k)}>{p.label}</button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={calibrationCode} title="calibration/calibration.ts" python={{ code: pythonCode, title: 'calibration/calibration.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">k1 <strong>{fmt(k.k1)}</strong>
            <input type="range" min={-0.6} max={0.6} step={0.01} value={k.k1} onChange={e => setK({ ...k, k1: Number(e.target.value) })} />
          </label>
          <label className="dl-slider">k2 <strong>{fmt(k.k2)}</strong>
            <input type="range" min={-0.6} max={0.6} step={0.01} value={k.k2} onChange={e => setK({ ...k, k2: Number(e.target.value) })} />
          </label>
          <label className="dl-slider">Board rotation <strong>{rotate}°</strong>
            <input type="range" min={-30} max={30} value={rotate} onChange={e => setRotate(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Tilt <strong>{tilt.toFixed(2)}</strong>
            <input type="range" min={-0.6} max={0.6} step={0.05} value={tilt} onChange={e => setTilt(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Board size <strong>{scale} px/square</strong>
            <input type="range" min={10} max={21} value={scale} onChange={e => setScale(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Detection noise <strong>{noise.toFixed(1)} px</strong>
            <input type="range" min={0} max={2} step={0.1} value={noise} onChange={e => setNoise(Number(e.target.value))} />
          </label>
        </div>
        <button type="button" className="btn btn-sm btn-primary" onClick={() => setResult({ calibration: calibrate(BOARD, observed, CAM), key })}>📐 Calibrate</button>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 New noise</button>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Photo through the lens</h3>
          <PixelCanvas image={photoImage} size={DISPLAY} pixelated={false} label="Checkerboard photographed through the distorting lens, with the detected corners">
            <svg className="cal-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
              <GridLines points={observed} className="cal-bent" />
              {observed.map((p, i) => <circle key={i} className="cal-corner" cx={p.x} cy={p.y} r={1.2} />)}
            </svg>
          </PixelCanvas>
          <p className="caption">Lines through the detected corners: bent by the lens</p>
        </div>
        <div className="demo-panel">
          <h3>{calibration ? 'Undistorted with the estimate' : 'Press Calibrate'}</h3>
          <PixelCanvas image={corrected} size={DISPLAY} pixelated={false} placeholderAspect={W / H}
            placeholder={<span className="muted">Estimate k1, k2 from the corners</span>} label="The photo undistorted with the estimated lens model">
            {straight && (
              <svg className="cal-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
                <GridLines points={straight} className="cal-straight" />
              </svg>
            )}
          </PixelCanvas>
          {calibration && <p className="caption">Straight lines of the fitted board pose: the squares follow them</p>}
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel cal-results">
          <h3>Estimate</h3>
          {calibration ? (
            <table className="cal-table">
              <thead><tr><th scope="col">parameter</th><th scope="col">true</th><th scope="col">estimated</th></tr></thead>
              <tbody>
                <tr><th>k1</th><td>{fmt(k.k1)}</td><td>{fmt(calibration.k.k1)}</td></tr>
                <tr><th>k2</th><td>{fmt(k.k2)}</td><td>{fmt(calibration.k.k2)}</td></tr>
                <tr><th>RMS error</th><td colSpan={2}>{calibration.rms[0].toFixed(2)} px without distortion → {calibration.rms[calibration.rms.length - 1].toFixed(2)} px after {calibration.rms.length - 1} steps</td></tr>
              </tbody>
            </table>
          ) : <p className="muted">{BOARD.length} corners detected, known board layout ({COLS}×{ROWS} inner corners).</p>}
        </div>
        <div className="demo-panel cal-chart">
          <h3>Radial shift</h3>
          <LineChart title="How far the lens moves a point, against its distance from the centre"
            series={[
              { label: 'true lens', color: 'var(--series-blue)', values: shifts(k) },
              ...(calibration ? [{ label: 'estimate', color: 'var(--series-orange)', values: shifts(calibration.k), dashed: true }] : []),
            ]}
            xLabel="distance from the centre (px)" xTickLabel={i => radii[i].toFixed(0)} formatY={v => v.toFixed(1)} height={170} />
          <p className="caption">Only radii the board covers are measured: beyond them the estimate is an extrapolation. A smaller board
          or more noise lets k1 and k2 trade off against each other, which is why real calibration uses many photos.</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Straight lines should stay straight</h4>
        <p>A pinhole camera maps every straight line in the world to a straight line in the image. Real lenses, especially
        wide-angle ones, bend lines that do not pass through the centre. The usual model moves each point along its radius,
        in normalised coordinates x = (u − cx) / f:</p>
        <p className="formula"><code>x_d = x · (1 + k1·r² + k2·r⁴),   r² = x² + y²</code></p>
        <p>With k1 &lt; 0 the image is squeezed towards the corners (<strong>barrel</strong>); with k1 &gt; 0 they are pulled
        out (<strong>pincushion</strong>); opposite signs of k1 and k2 give the wavy <strong>moustache</strong>.</p>
        <p>To <strong>calibrate</strong>, photograph a checkerboard: its corners are easy to detect precisely and their layout
        is known. For a guess of (k1, k2), undistort the detected corners: a flat board seen by a pinhole camera is related to
        its layout by a <strong>homography</strong>, so fit one by least squares. Push the board back through that
        homography and the lens model, and compare with the detections: that <strong>reprojection error</strong> is what
        Gauss–Newton minimises over k. Undistorting the photo is then a remap: every output pixel looks up where the lens put
        it.</p>
        <p>This lesson knows the focal length and the image centre. Full calibration (Zhang's method, used by OpenCV) estimates
        them too, from several photos of the board at different angles: a single fronto-parallel view cannot separate the
        focal length from the board's distance.</p>
      </div>
    </div>
  );
}
