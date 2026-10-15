import { useEffect, useMemo, useState } from 'react';
import watershedCode from './watershed.ts?raw';
import pythonCode from './watershed.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { rgbImage } from '../../lib/image';
import { seededRandom } from '../../lib/random';
import { hotColor } from '../hough/hough';
import { labelColour, labelComponents } from '../regions/components';
import { distanceTransform, findMarkers, randomDiscs, rasteriseDiscs, watershed } from './watershed';
import './watershed.css';

const W = 220, H = 150;
const DISPLAY = 330;
const STEPS = 60;

export function WatershedSection() {
  const [count, setCount] = useState(10);
  const [overlap, setOverlap] = useState(0.4);
  const [seed, setSeed] = useState(4);
  const [radius, setRadius] = useState(6);
  const [minDistance, setMinDistance] = useState(4);
  const [level, setLevel] = useState(STEPS);
  const [playing, setPlaying] = useState(false);
  const [showCode, setShowCode] = useState(false);

  const animating = playing && level < STEPS;
  useEffect(() => {
    if (!animating) return;
    const id = setInterval(() => setLevel(l => Math.min(STEPS, l + 1)), 60);
    return () => clearInterval(id);
  }, [animating]);

  const data = useMemo(() => {
    const discs = randomDiscs(W, H, count, overlap, seededRandom(seed));
    const mask = rasteriseDiscs(discs, W, H);
    const dist = distanceTransform(mask, W, H);
    return { discs, mask, dist, maxDist: Math.max(...dist), components: labelComponents(mask, W, H, 4) };
  }, [count, overlap, seed]);

  const result = useMemo(() => {
    const { markers, count: markerCount } = findMarkers(data.dist, W, H, radius, minDistance);
    return { markers, markerCount, ...watershed(data.dist.map(v => -v), markers, data.mask, W, H) };
  }, [data, radius, minDistance]);

  const images = useMemo(() => ({
    mask: rgbImage(W, H, i => (data.mask[i] ? labelColour(data.components.labels[i]) : [18, 18, 24])),
    dist: rgbImage(W, H, i => (data.mask[i] ? hotColor(data.dist[i] / data.maxDist) : [18, 18, 24])),
  }), [data]);

  // One dot per marker, at the mean position of its pixels.
  const markerDots = useMemo(() => {
    const sums = Array.from({ length: result.markerCount + 1 }, () => [0, 0, 0]);
    result.markers.forEach((m, i) => { if (m) { sums[m][0] += i % W; sums[m][1] += Math.floor(i / W); sums[m][2]++; } });
    return sums.slice(1).map(([x, y, n]) => ({ x: x / n + 0.5, y: y / n + 0.5 }));
  }, [result]);

  const flood = useMemo(() => {
    const waterline = -data.maxDist * (1 - level / STEPS);           // landscape height reached by the water
    return rgbImage(W, H, i => {
      if (!data.mask[i]) return [18, 18, 24];
      if (result.floodedAt[i] > waterline + 1e-9) return hotColor(0.12 + 0.5 * (data.dist[i] / data.maxDist));
      if (result.lines[i]) return [255, 255, 255];
      return labelColour(result.labels[i]);
    });
  }, [data, result, level]);

  const objects = result.markerCount;
  const verdict = objects === data.discs.length ? 'exactly right' : objects > data.discs.length ? 'over-segmented' : 'under-segmented';

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <label className="dl-slider">Coins <strong>{count}</strong>
          <input type="range" min={2} max={20} value={count} onChange={e => setCount(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Overlap <strong>{Math.round(overlap * 100)}%</strong>
          <input type="range" min={0} max={0.8} step={0.05} value={overlap} onChange={e => setOverlap(Number(e.target.value))} />
        </label>
        <button type="button" className="btn btn-sm" onClick={() => setSeed(s => s + 1)}>🎲 New coins</button>
        <span className="ml-auto">
          <ShowCodeButton code={watershedCode} title="watershed/watershed.ts" python={{ code: pythonCode, title: 'watershed/watershed.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      <div className="demo-controls">
        <label className="dl-slider">Marker spacing <strong>{radius} px</strong>
          <input type="range" min={1} max={20} value={radius} onChange={e => setRadius(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Min. marker depth <strong>{minDistance} px</strong>
          <input type="range" min={1} max={15} value={minDistance} onChange={e => setMinDistance(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Water level <strong>{Math.round((level / STEPS) * 100)}%</strong>
          <input type="range" min={0} max={STEPS} value={level} onChange={e => { setPlaying(false); setLevel(Number(e.target.value)); }} />
        </label>
        <button type="button" className="btn btn-sm btn-primary" onClick={() => { setLevel(0); setPlaying(true); }}>▶ Flood</button>
      </div>

      <div className="ws-grid">
        <div className="demo-panel">
          <h3>Threshold + connected components</h3>
          <PixelCanvas image={images.mask} size={DISPLAY} label="Binary coins, coloured by connected component" />
          <p className="caption">{data.components.count} components for {data.discs.length} coins: touching coins merge.</p>
        </div>
        <div className="demo-panel">
          <h3>Distance to background + markers</h3>
          <PixelCanvas image={images.dist} size={DISPLAY} label="Distance transform, brighter is further from the background; markers in cyan">
            <svg className="ws-markers" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
              {markerDots.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={2.2} />)}
            </svg>
          </PixelCanvas>
          <p className="caption">One peak per coin; the cyan markers are the water sources.</p>
        </div>
        <div className="demo-panel">
          <h3>Flooding from the markers</h3>
          <PixelCanvas image={flood} size={DISPLAY} label="Basins flooded so far, with the white dams where two floods meet" />
          <p className="caption">Each basin fills from its marker; white = dams where two meet.</p>
        </div>
      </div>

      <div className="dl-stat-row">
        <span>Coins <strong>{data.discs.length}</strong></span>
        <span>Connected components <strong>{data.components.count}</strong></span>
        <span>Watershed objects <strong>{objects}</strong> ({verdict})</span>
      </div>

      <div className="demo-explanation">
        <h4>Splitting touching objects</h4>
        <p>Threshold a picture of coins, cells or grains and the ones that touch fuse into one blob, so counting connected components
        undercounts. The <strong>distance transform</strong> gives each foreground pixel its distance to the nearest background
        pixel: it peaks in the middle of each coin and dips at the narrow neck where two coins touch.</p>
        <p>Turn it upside down and it is a landscape with one valley per coin. The <strong>watershed</strong> floods it: put a water
        source (a <strong>marker</strong>) in each valley and raise the water level. Each pixel joins the first flood that reaches it,
        and where two floods meet we build a dam. The dams are the cut lines. Press ▶ Flood to watch.</p>
        <p>Everything depends on the markers. With every local maximum as a marker (spacing 1 px), the slightly bumpy distance map
        gives too many sources and coins get chopped up: <strong>over-segmentation</strong>. With a large spacing, two coins share
        one marker and stay merged. Strongly overlapping coins have no neck to cut at all. The same idea, flooding a gradient
        image, segments photos too, which is how watershed is used in microscopy.</p>
      </div>
    </div>
  );
}
