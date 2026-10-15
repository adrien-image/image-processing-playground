import { useMemo, useState } from 'react';
import kmeansCode from './kmeans.ts?raw';
import otsuCode from './otsu.ts?raw';
import kmeansPy from './kmeans.py?raw';
import otsuPy from './otsu.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { checkerboard, makeCanvas } from '../../lib/demoImages';
import { grayToImageData } from '../../lib/image';
import { toImageData } from '../../lib/canvas';
import { applyThreshold, displayHistogram, histogram256, otsuThreshold } from './otsu';
import { clusterCss, kmeans, labelsToColourImage, labelsToImage } from './kmeans';
import { HistogramView } from './HistogramView';
import './segmentation.css';

type DemoImage = 'bimodal' | 'tricluster' | 'gradient' | 'checkerboard';
type Method = 'otsu' | 'kmeans-2' | 'kmeans-3';

const SIZE = 64;
const DISPLAY = 300;

const DEMOS: Record<DemoImage, { label: string; make: () => HTMLCanvasElement }> = {
  bimodal: {
    label: 'Two‑tone',
    make: () => makeCanvas(SIZE, SIZE, ctx => {
      ctx.fillStyle = '#ddd';
      ctx.fillRect(0, 0, SIZE, SIZE);
      ctx.fillStyle = '#222';
      ctx.beginPath(); ctx.arc(SIZE * 0.3, SIZE * 0.3, SIZE * 0.18, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(SIZE * 0.7, SIZE * 0.7, SIZE * 0.15, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(SIZE * 0.7, SIZE * 0.25, SIZE * 0.1, 0, Math.PI * 2); ctx.fill();
    }),
  },
  tricluster: {
    label: 'Three‑band',
    make: () => makeCanvas(SIZE, SIZE, ctx => {
      const bandW = SIZE / 3;
      ctx.fillStyle = '#222'; ctx.fillRect(0, 0, bandW, SIZE);
      ctx.fillStyle = '#888'; ctx.fillRect(bandW, 0, bandW, SIZE);
      ctx.fillStyle = '#ddd'; ctx.fillRect(bandW * 2, 0, bandW, SIZE);
    }),
  },
  gradient: {
    label: 'Gradient',
    make: () => makeCanvas(SIZE, SIZE, ctx => {
      const gradient = ctx.createLinearGradient(0, 0, SIZE, 0);
      gradient.addColorStop(0, '#000');
      gradient.addColorStop(1, '#fff');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, SIZE, SIZE);
    }),
  },
  checkerboard: { label: 'Checkerboard', make: () => checkerboard(SIZE, 8) },
};

const METHODS: { key: Method; label: string }[] = [
  { key: 'otsu', label: 'Otsu Threshold' },
  { key: 'kmeans-2', label: 'K‑means (k=2)' },
  { key: 'kmeans-3', label: 'K‑means (k=3)' },
];

export function SegmentationSection() {
  const [showCode, setShowCode] = useState(false);
  const [demo, setDemo] = useState<DemoImage | 'upload'>('bimodal');
  const [source, setSource] = useState<ImageSource>(() => DEMOS.bimodal.make());
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [method, setMethod] = useState<Method>('otsu');
  // null = follow Otsu's suggestion; a number = the user moved the slider.
  const [userThreshold, setUserThreshold] = useState<number | null>(null);

  const { image, error: loadError } = useImageData(source, 128);

  const analysis = useMemo(() => {
    if (!image) return null;
    const hist = histogram256(image.gray);
    return { hist, display: displayHistogram(hist), otsu: otsuThreshold(hist) };
  }, [image]);

  const k = method === 'kmeans-3' ? 3 : 2;
  const km = useMemo(
    () => (image && method !== 'otsu' ? kmeans(image.gray, k) : null),
    [image, method, k],
  );

  const threshold = userThreshold ?? analysis?.otsu ?? 128;

  const segmented = useMemo(() => {
    if (!image) return null;
    const { gray, width, height } = image;
    if (km) return grayToImageData(labelsToImage(km.labels, km.centroids), width, height);
    const mask = applyThreshold(gray, threshold);
    return grayToImageData(Float64Array.from(mask, v => v * 255), width, height);
  }, [image, km, threshold]);

  const clusters = useMemo(
    () => (image && km ? toImageData(labelsToColourImage(km.labels, image.width, image.height), image.width, image.height) : null),
    [image, km],
  );

  const pickDemo = (name: DemoImage) => {
    setDemo(name);
    setSource(DEMOS[name].make());
    setUserThreshold(null);
    setUploadError(null);
  };

  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Demo image:</span>
        {(Object.keys(DEMOS) as DemoImage[]).map(name => (
          <button key={name} type="button" aria-pressed={demo === name}
            className={`btn btn-sm${demo === name ? ' btn-primary' : ''}`} onClick={() => pickDemo(name)}>
            {DEMOS[name].label}
          </button>
        ))}
        <ImageUploader
          label="📁 Upload"
          onImage={(src) => { setDemo('upload'); setSource(src); setUserThreshold(null); setUploadError(null); }}
          onError={setUploadError}
        />
      </div>

      <div className="demo-controls">
        <span className="demo-label">Method:</span>
        {METHODS.map(m => (
          <button key={m.key} type="button" aria-pressed={method === m.key}
            className={`btn btn-sm${method === m.key ? ' btn-primary' : ''}`} onClick={() => setMethod(m.key)}>
            {m.label}
          </button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton
            code={method === 'otsu' ? otsuCode : kmeansCode}
            title={method === 'otsu' ? 'segmentation/otsu.ts' : 'segmentation/kmeans.ts'}
            python={method === 'otsu'
              ? { code: otsuPy, title: 'segmentation/otsu.py' }
              : { code: kmeansPy, title: 'segmentation/kmeans.py' }}
            open={showCode}
            onOpenChange={setShowCode}
          />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} label="Original image" />
        </div>
        <div className="demo-panel">
          <h3>{method === 'otsu' ? `Binary (value > ${threshold})` : `K‑means (k=${k}), centroid values`}</h3>
          <PixelCanvas image={segmented} size={DISPLAY} label="Segmentation result" />
        </div>
        {method !== 'otsu' && (
          <div className="demo-panel">
            <h3>Clusters (coloured)</h3>
            <PixelCanvas image={clusters} size={DISPLAY} label="Cluster map" />
          </div>
        )}
      </div>

      {method === 'otsu' && analysis && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="seg-threshold">
              Threshold: <strong>{threshold}</strong> <span className="muted">(value &gt; t is foreground)</span>
            </label>
            <input id="seg-threshold" type="range" min={0} max={254} step={1} value={threshold}
              onChange={(e) => setUserThreshold(Number(e.target.value))} className="demo-slider" />
            <button type="button" className="otsu-optimal" onClick={() => setUserThreshold(null)}
              aria-label={`Reset threshold to Otsu's value, ${analysis.otsu}`}>
              ⬤ Otsu: {analysis.otsu}
            </button>
          </div>
        </div>
      )}

      {km && (
        <div className="demo-slider-section">
          <div className="otsu-km-info">
            <span className="demo-label">Cluster centres:</span>
            {Array.from(km.centroids).map((c, i) => (
              <span key={i} className="otsu-km-centroid" style={{ borderColor: clusterCss(i) }}>
                <span className="otsu-km-swatch" style={{ background: clusterCss(i) }} />
                C{i}: {c.toFixed(1)}
              </span>
            ))}
            <span className="muted">· converged in {km.iterations} iteration{km.iterations === 1 ? '' : 's'}</span>
          </div>
        </div>
      )}

      {analysis && (
        <HistogramView
          histogram={analysis.display}
          threshold={method === 'otsu' ? threshold : undefined}
          otsuThreshold={method === 'otsu' ? analysis.otsu : undefined}
          label={`Gray-level histogram${method === 'otsu' ? `, threshold ${threshold}, Otsu ${analysis.otsu}` : ''}`}
        />
      )}

      <div className="demo-explanation">
        {method === 'otsu' ? (
          <>
            <h4>How Otsu's Method Works</h4>
            <p>Otsu's method finds the optimal threshold by <strong>maximising between-class variance</strong>.
            It tries all 255 possible cuts of the 256-level histogram and keeps the one that best separates the two groups.</p>
            <div className="formula"><code>σ²<sub>between</sub>(t) = ω₁(t) · ω₂(t) · [μ₁(t) − μ₂(t)]²</code></div>
            <p>Works well for <strong>bimodal</strong> images (two distinct peaks). Struggles with images having more than two intensity populations.</p>
          </>
        ) : (
          <>
            <h4>How K‑means Clustering Works</h4>
            <p>K‑means partitions the pixels into <strong>k clusters</strong> by minimising the distance from each pixel to its cluster centre.
            The algorithm alternates between assigning pixels to the nearest centre and updating centres to the mean of their assigned pixels.</p>
            <div className="formula">
              <code>1. Assign: cᵢ = argmin‖xᵢ − μⱼ‖&emsp;2. Update: μⱼ = (1/|Cⱼ|) Σ xᵢ</code>
            </div>
            <p>Unlike Otsu, K‑means can separate <strong>more than two</strong> intensity levels. Try the <strong>Three‑band</strong> image with k=3 to see it in action.
            With k=2 on three bands, it merges the two closest bands — a common failure mode.</p>
          </>
        )}
        {demo === 'gradient' && (
          <p className="note-warning">⚠ The gradient has a flat histogram with no peaks — there is no natural boundary, so any threshold is as good as another.</p>
        )}
        {demo === 'checkerboard' && (
          <p className="note-warning">⚠ The checkerboard has alternating black and white squares — thresholding separates them but the result is a checkerboard pattern, not a meaningful segmentation.</p>
        )}
        {demo === 'tricluster' && method === 'otsu' && (
          <p className="note-warning">⚠ The three‑band image has <strong>three</strong> distinct intensity levels. Otsu can only find one threshold, so it merges two bands into one. K‑means with k=3 separates all three correctly.</p>
        )}
      </div>
    </div>
  );
}
