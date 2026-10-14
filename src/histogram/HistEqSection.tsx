import { useMemo, useState } from 'react';
import equalizeCode from './equalize.ts?raw';
import { ShowCodeButton } from '../components/CodeViewer';
import { PixelCanvas } from '../components/PixelCanvas';
import { ImageUploader } from '../components/ImageUploader';
import { useImageData, type ImageSource } from '../hooks/useImageData';
import { makeCanvas } from '../lib/demoImages';
import { grayToImageData } from '../lib/image';
import { displayHistogram, histogram256 } from '../segmentation/otsu';
import { HistogramView } from '../segmentation/HistogramView';
import { equalize } from './equalize';
import { MappingCurve } from './MappingCurve';
import './histogram.css';

/** Low-contrast demo: patches crowded into the dark end of the range. */
function testImage(): HTMLCanvasElement {
  return makeCanvas(100, 100, ctx => {
    ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, 100, 100);
    const levels: [number, number, number, number, number][] = [
      [20, 20, 60, 60, 100], [12, 12, 30, 30, 60],
      [50, 50, 35, 35, 140], [70, 12, 20, 20, 200],
      [15, 55, 25, 25, 90], [55, 55, 30, 15, 170],
      [72, 60, 18, 25, 220], [10, 70, 15, 15, 50],
      [45, 12, 15, 18, 180], [60, 75, 30, 15, 130],
      [35, 72, 12, 12, 240], [80, 80, 15, 15, 75],
    ];
    for (const [x, y, w, h, level] of levels) {
      ctx.fillStyle = `rgb(${level},${level},${level})`;
      ctx.fillRect(x, y, w, h);
    }
    const grad = ctx.createRadialGradient(50, 60, 5, 50, 60, 20);
    grad.addColorStop(0, '#eee');
    grad.addColorStop(1, '#333');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(50, 60, 20, 0, Math.PI * 2); ctx.fill();
  });
}

const DISPLAY = 260;

export function HistEqSection() {
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(testImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 512);

  const result = useMemo(() => {
    if (!image) return null;
    const { output, mapping } = equalize(image.gray);
    return {
      mapping,
      before: displayHistogram(histogram256(image.gray)),
      after: displayHistogram(histogram256(output)),
      original: grayToImageData(image.gray, image.width, image.height),
      equalised: grayToImageData(output, image.width, image.height),
    };
  }, [image]);

  const error = uploadError ?? loadError;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Image:</span>
        <ImageUploader onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} label="📁 Upload photo" />
        {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(testImage()); setIsDemo(true); }}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={equalizeCode} title="histogram/equalize.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>{isDemo ? 'Original (low-contrast demo)' : 'Original (grayscale)'}</h3>
          <PixelCanvas image={result?.original ?? null} size={DISPLAY} pixelated={isDemo} label="Original grayscale image" />
        </div>
        <div className="demo-panel">
          <h3>Equalised</h3>
          <PixelCanvas image={result?.equalised ?? null} size={DISPLAY} pixelated={isDemo} label="Equalised image" />
        </div>
      </div>

      {result && (
        <div className="demo-slider-section row-center">
          <div>
            <div className="panel-heading">Original histogram</div>
            <HistogramView histogram={result.before} width={280} height={140} label="Histogram of the original image" />
          </div>
          <div>
            <div className="panel-heading">Mapping (input → output)</div>
            <MappingCurve mapping={result.mapping} size={140} />
          </div>
          <div>
            <div className="panel-heading">Equalised histogram</div>
            <HistogramView histogram={result.after} width={280} height={140} label="Histogram of the equalised image" />
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Histogram Equalisation</h4>
        <p>Histogram equalisation spreads out the most frequent intensity values to improve contrast.
        It computes the <strong>Cumulative Distribution Function (CDF)</strong> of the pixel intensities,
        then maps each pixel to its percentile rank × 255.</p>
        <p>The CDF is normalised so the lowest occupied intensity maps to 0 and the highest to 255,
        stretching the output to the full dynamic range. The <strong>mapping</strong> curve is that normalised CDF: steep where many pixels share similar levels (they get pulled apart), flat where few pixels are.</p>
        <p>Compare the two histograms: crowded levels are spread across 0–255. The bars don't become perfectly flat, because pixels that share a level must stay together.</p>
      </div>
    </div>
  );
}
