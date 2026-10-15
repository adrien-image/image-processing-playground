import { useMemo, useState } from 'react';
import slicCode from './slic.ts?raw';
import pythonCode from './slic.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { ImageUploader } from '../../components/ImageUploader';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { toImageData } from '../../lib/canvas';
import { scene } from '../../lib/demoImages';
import { boundaries, meanColours, slic } from './slic';
import './superpixels.css';

const DISPLAY = 340;

export function SuperpixelsSection() {
  const [source, setSource] = useState<ImageSource>(() => scene(256));
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [k, setK] = useState(200);
  const [compactness, setCompactness] = useState(10);
  const [iterations, setIterations] = useState(10);
  const [showCentres, setShowCentres] = useState(true);
  const [showCode, setShowCode] = useState(false);
  const { image, error: loadError } = useImageData(source, 256);

  const result = useMemo(() => (image ? slic(image.imageData.data, image.width, image.height, k, compactness, iterations) : null), [image, k, compactness, iterations]);

  const views = useMemo(() => {
    if (!image || !result) return null;
    const { width, height } = image;
    const edge = boundaries(result.labels, width, height);
    const overlay = new Uint8ClampedArray(image.imageData.data);
    if (iterations > 0) for (let i = 0; i < edge.length; i++) if (edge[i]) overlay.set([255, 230, 60], i * 4);
    const mean = iterations > 0 ? meanColours(image.imageData.data, result.labels, result.count) : new Uint8ClampedArray(image.imageData.data);
    return { overlay: toImageData(overlay, width, height), mean: toImageData(mean, width, height) };
  }, [image, result, iterations]);

  const error = uploadError ?? loadError;
  const centres = image && result && showCentres && (
    <svg className="sp-centres" viewBox={`0 0 ${image.width} ${image.height}`} preserveAspectRatio="none" aria-hidden="true">
      {result.centres.map((c, i) => <circle key={i} cx={c.x + 0.5} cy={c.y + 0.5} r={1.4} />)}
    </svg>
  );

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <label className="dl-slider">Superpixels K <strong>{k}</strong>
          <input type="range" min={20} max={800} step={10} value={k} onChange={e => setK(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Compactness m <strong>{compactness}</strong>
          <input type="range" min={1} max={60} value={compactness} onChange={e => setCompactness(Number(e.target.value))} />
        </label>
        <label className="dl-slider">Iterations <strong>{iterations}</strong>
          <input type="range" min={0} max={15} value={iterations} onChange={e => setIterations(Number(e.target.value))} />
        </label>
        <label className="dl-slider"><input type="checkbox" checked={showCentres} onChange={e => setShowCentres(e.target.checked)} /> Show centres</label>
        <ImageUploader label="📁 Upload" onImage={src => { setSource(src); setUploadError(null); }} onError={setUploadError} />
        <span className="ml-auto">
          <ShowCodeButton code={slicCode} title="superpixels/slic.ts" python={{ code: pythonCode, title: 'superpixels/slic.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {views && result && image && (
        <div className="sp-grid">
          <div className="demo-panel">
            <h3>{iterations === 0 ? 'Initial grid of centres' : 'Superpixel boundaries'}</h3>
            <PixelCanvas image={views.overlay} size={DISPLAY} label="The image with superpixel boundaries drawn in yellow">{centres}</PixelCanvas>
            <p className="caption">{iterations === 0 ? 'Centres start on a grid, moved off edges.' : 'Boundaries snap to colour edges.'}</p>
          </div>
          <div className="demo-panel">
            <h3>Each superpixel's mean colour</h3>
            <PixelCanvas image={views.mean} size={DISPLAY} label="Each superpixel filled with its mean colour" />
            <p className="caption">{iterations === 0 ? 'Run at least one iteration.' : `${result.count} regions stand in for ${(image.width * image.height).toLocaleString()} pixels.`}</p>
          </div>
        </div>
      )}

      <div className="dl-stat-row">
        <span>Grid spacing S <strong>{result ? result.step.toFixed(1) : '—'} px</strong></span>
        <span>Regions <strong>{result?.count ?? 0}</strong></span>
      </div>

      <div className="demo-explanation">
        <h4>Superpixels</h4>
        <p>Many algorithms are slow per pixel but fine per region. <strong>Superpixels</strong> group pixels into a few hundred small,
        roughly equal regions that respect colour edges, so a later step (a graph cut, a classifier, a tracker) can work on regions
        instead of pixels.</p>
        <p><strong>SLIC</strong> is k-means on (L, a, b, x, y): CIELAB colour plus position. Centres start on a grid with spacing
        S = √(N / K); each iteration gives every pixel to the closest centre and moves each centre to the mean of its pixels. Each
        centre only looks at a 2S × 2S window around itself, which makes an iteration cost O(N) whatever K is. Set the iterations to 0
        and step up to watch the grid bend towards the edges; it settles in about 10.</p>
        <p className="formula"><code>D = √( d_lab² + (d_xy / S)² · m² )</code></p>
        <p>The <strong>compactness</strong> m weighs distance against colour. A small m lets regions wrap tightly around edges but
        makes them ragged; a large m makes them square tiles that ignore the image. Finally, small pieces cut off from their
        centre are merged into a neighbour so that every superpixel is one connected region.</p>
      </div>
    </div>
  );
}
