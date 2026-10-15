import { useEffect, useMemo, useState } from 'react';
import stereoCode from './stereo.ts?raw';
import pythonCode from './stereo.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { grayToImageData, rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { hotColor } from '../hough/hough';
import { blockMatch, demoLayers, leftRightCheck, renderStereo, score } from './stereo';
import './stereo.css';

const W = 240, H = 160;
const DISPLAY = 330;

/** Disparity as a heat colour (brighter = closer); rejected pixels gray. */
const colour = (d: number): readonly [number, number, number] => (d < 0 ? [70, 70, 80] : hotColor(0.15 + 0.85 * (d / 24)));

export function StereoSection() {
  const [radius, setRadius] = useState(3);
  const [maxD, setMaxD] = useState(28);
  const [noise, setNoise] = useState(4);
  const [lrCheck, setLrCheck] = useState(true);
  const [flick, setFlick] = useState(true);
  const [showRight, setShowRight] = useState(false);
  const [showCode, setShowCode] = useState(false);

  useEffect(() => {
    if (!flick) return;
    const id = setInterval(() => setShowRight(s => !s), 500);
    return () => clearInterval(id);
  }, [flick]);

  const pair = useMemo(() => renderStereo(W, H, demoLayers(W, H), noise, seededRandom(3)), [noise]);
  const result = useMemo(() => {
    const left = blockMatch(pair.left, pair.right, W, H, maxD, radius);
    const estimate = lrCheck ? leftRightCheck(left, blockMatch(pair.left, pair.right, W, H, maxD, radius, true), W, H) : left;
    return { estimate, stats: score(estimate, pair, maxD) };
  }, [pair, maxD, radius, lrCheck]);

  const images = useMemo(() => ({
    left: grayToImageData(pair.left, W, H),
    right: grayToImageData(pair.right, W, H),
    truth: rgbImage(W, H, i => colour(pair.disparity[i])),
    estimate: rgbImage(W, H, i => colour(result.estimate[i])),
    errors: rgbImage(W, H, i => {
      const x = i % W;
      if (x < maxD) return [30, 30, 36];                                     // no full disparity range at the left edge
      if (pair.occluded[i]) return result.estimate[i] < 0 ? [70, 120, 255] : [150, 90, 220];
      if (result.estimate[i] < 0) return [70, 70, 80];
      return Math.abs(result.estimate[i] - pair.disparity[i]) > 1 ? [248, 113, 113] : [24, 60, 40];
    }),
  }), [pair, result, maxD]);

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <label className="dl-slider">Window <strong>{2 * radius + 1}×{2 * radius + 1}</strong>
          <input type="range" min={0} max={10} value={radius} onChange={e => setRadius(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Max disparity <strong>{maxD} px</strong>
          <input type="range" min={10} max={40} value={maxD} onChange={e => setMaxD(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Noise <strong>{noise}</strong>
          <input type="range" min={0} max={30} value={noise} onChange={e => setNoise(Number(e.target.value))} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={lrCheck} onChange={e => setLrCheck(e.target.checked)} /> Left–right check</label>
        <label className="dl-slider"><input type="checkbox" checked={flick} onChange={e => setFlick(e.target.checked)} /> Flick left / right</label>
        <span className="ml-auto">
          <ShowCodeButton code={stereoCode} title="stereo/stereo.ts" python={{ code: pythonCode, title: 'stereo/stereo.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="stereo-grid">
        <div className="demo-panel">
          <h3>{showRight && flick ? 'Right camera' : 'Left camera'}</h3>
          <PixelCanvas image={showRight && flick ? images.right : images.left} size={DISPLAY} label="Stereo image" />
          <p className="caption">Near objects jump further between the views.</p>
        </div>
        <div className="demo-panel">
          <h3>True disparity</h3>
          <PixelCanvas image={images.truth} size={DISPLAY} label="True disparity: brighter is closer" />
          <p className="caption">Brighter = larger disparity = closer.</p>
        </div>
        <div className="demo-panel">
          <h3>Block matching</h3>
          <PixelCanvas image={images.estimate} size={DISPLAY} label="Estimated disparity" />
          <p className="caption">Gray = rejected (no reliable match).</p>
        </div>
        <div className="demo-panel">
          <h3>Errors</h3>
          <PixelCanvas image={images.errors} size={DISPLAY} label="Where the estimate is wrong" />
          <ul className="stereo-legend">
            <li><i style={{ background: '#185c3c' }} />correct (±1 px)</li>
            <li><i style={{ background: '#f87171' }} />wrong</li>
            <li><i style={{ background: '#4678ff' }} />occluded, rejected</li>
            <li><i style={{ background: '#965adc' }} />occluded, guessed</li>
          </ul>
        </div>
      </div>

      <div className="dl-stat-row">
        <span>Wrong by &gt; 1 px <strong>{(result.stats.bad * 100).toFixed(1)}%</strong> of matched visible pixels</span>
        <span>Rejected <strong>{(result.stats.invalid * 100).toFixed(1)}%</strong></span>
        <span>Occlusions caught <strong>{(result.stats.occlusionsCaught * 100).toFixed(0)}%</strong></span>
      </div>

      <div className="demo-explanation">
        <h4>Depth from two views</h4>
        <p>Hold a finger in front of your face and close each eye in turn: it jumps more than the background. Two cameras a distance B
        apart see a point at depth Z shifted by a <strong>disparity</strong> d = f·B / Z pixels (f is the focal length in pixels), so
        measuring d gives the depth. After rectification the shift is purely horizontal, so the search runs along one row.</p>
        <p className="formula"><code>Z = f · B / d</code></p>
        <p><strong>Block matching</strong> compares a window around each left pixel with windows in the right image d pixels to the left, for every
        d, and keeps the best. Shrink the window to 1×1 and the estimate is mostly noise: one pixel matches many places. Grow it and the
        noise goes, but depth edges spread: a window straddling the box and the wall gets the box's disparity ("foreground fattening").</p>
        <p>Some pixels are <strong>occluded</strong>: next to a near object, part of the background is visible to one camera only, so it has no
        correct match. The <strong>left–right check</strong> matches in both directions and rejects pixels where they disagree; most occluded
        pixels are caught. Modern methods add smoothness between neighbouring pixels (semi-global matching) or learn the matching
        with CNNs.</p>
      </div>
    </div>
  );
}
