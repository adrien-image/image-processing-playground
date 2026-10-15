import { useMemo, useState } from 'react';
import houghCode from './hough.ts?raw';
import pythonCode from './hough.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { grayToImageData, rgbImage } from '../../lib/image';
import { cannyWithGradients } from '../edges/edgeDetectors';
import { hotColor, houghCircles, houghLines } from './hough';
import './hough.css';

function testImage(): HTMLCanvasElement {
  return makeCanvas(200, 200, ctx => {
    ctx.fillStyle = '#222'; ctx.fillRect(0, 0, 200, 200);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(30, 30); ctx.lineTo(160, 120); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(40, 140); ctx.lineTo(170, 60); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(90, 20); ctx.lineTo(80, 170); ctx.stroke();
    ctx.strokeRect(40, 40, 120, 120);
    ctx.beginPath(); ctx.arc(140, 140, 30, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(75, 85, 18, 0, Math.PI * 2); ctx.stroke();
  });
}

const DISPLAY = 220;

export function HoughSection() {
  const [lineThresh, setLineThresh] = useState(40);
  const [minLen, setMinLen] = useState(30);
  const [circThresh, setCircThresh] = useState(45);
  const [minR, setMinR] = useState(10);
  const [maxR, setMaxR] = useState(40);
  const [showCode, setShowCode] = useState(false);
  const [source, setSource] = useState<ImageSource>(testImage);
  const [isDemo, setIsDemo] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 240);

  // Stage 1: thin edges + gradients (only when the image changes).
  const edgeData = useMemo(
    () => (image ? cannyWithGradients(image.gray, image.width, image.height, 15, 40) : null),
    [image],
  );

  // Stage 2a: lines — line sliders don't recompute circles, and vice versa.
  const lineResult = useMemo(
    () => (image && edgeData ? houghLines(edgeData.edges, image.width, image.height, { threshold: lineThresh, minLength: minLen }) : null),
    [image, edgeData, lineThresh, minLen],
  );

  // Stage 2b: circles.
  const circleResult = useMemo(
    () => (image && edgeData
      ? houghCircles(edgeData.edges, edgeData.gx, edgeData.gy, image.width, image.height, { minR, maxR, threshold: circThresh / 100 })
      : null),
    [image, edgeData, minR, maxR, circThresh],
  );

  const edgeImage = useMemo(
    () => (image && edgeData ? grayToImageData(edgeData.edges, image.width, image.height) : null),
    [image, edgeData],
  );

  // Line accumulator, drawn with ρ horizontal and θ vertical.
  const lineSpace = useMemo(() => {
    if (!lineResult) return null;
    const { accumulator, numTheta, numRho } = lineResult;
    let max = 0;
    for (const v of accumulator) if (v > max) max = v;
    return rgbImage(numRho, numTheta, i => {
      const t = Math.floor(i / numRho), r = i % numRho;
      return hotColor(max ? Math.sqrt(accumulator[t * numRho + r] / max) : 0);
    });
  }, [lineResult]);

  const circleSpace = useMemo(() => {
    if (!circleResult || !image) return null;
    const { centreMap } = circleResult;
    return rgbImage(image.width, image.height, i => hotColor(Math.min(1, centreMap[i])));
  }, [circleResult, image]);

  const error = uploadError ?? loadError;
  const w = image?.width ?? 1, h = image?.height ?? 1;

  return (
    <div className="demo-section">
      <div className="hough-layout">
        <div className="hough-left demo-explanation">
          <div className="demo-controls hough-source">
            <ImageUploader label="📁 Upload" onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }} onError={setUploadError} />
            {!isDemo && <button type="button" className="btn btn-sm" onClick={() => { setSource(testImage()); setIsDemo(true); }}>Demo image</button>}
            <ShowCodeButton code={houghCode} title="hough/hough.ts" python={{ code: pythonCode, title: 'hough/hough.py' }} open={showCode} onOpenChange={setShowCode} />
          </div>
          {error && <div className="error-banner" role="alert">⚠ {error}</div>}

          <h3>Hough Line Transform</h3>
          <p>Each edge pixel votes for every line <strong>(ρ, θ)</strong> through it. Peaks in the accumulator are lines;
          each is then traced along the edges to find where the actual segment starts and ends.</p>
          <div className="demo-slider-section hough-sliders">
            <div className="hough-slider-row">
              <label htmlFor="hough-lt">Vote threshold: <strong>{lineThresh}</strong></label>
              <input id="hough-lt" type="range" min={10} max={150} value={lineThresh} onChange={e => setLineThresh(Number(e.target.value))} />
            </div>
            <div className="hough-slider-row">
              <label htmlFor="hough-ml">Min segment length: <strong>{minLen}</strong> px</label>
              <input id="hough-ml" type="range" min={5} max={180} value={minLen} onChange={e => setMinLen(Number(e.target.value))} />
            </div>
            <div className="hough-stat">Lines detected: <strong>{lineResult?.lines.length ?? 0}</strong></div>
          </div>

          <h3>Hough Circle Transform</h3>
          <p>Each edge pixel votes for the centres <strong>(a, b)</strong> lying a distance r along its gradient, for every radius r.
          A circle's score is its votes divided by its circumference — roughly the fraction of the circle that is visible.</p>
          <div className="demo-slider-section hough-sliders">
            <div className="hough-slider-row">
              <label htmlFor="hough-ct">Min score: <strong>{circThresh}%</strong></label>
              <input id="hough-ct" type="range" min={10} max={100} value={circThresh} onChange={e => setCircThresh(Number(e.target.value))} />
            </div>
            <div className="hough-slider-row">
              <label htmlFor="hough-minr">Min radius: <strong>{minR}</strong> px</label>
              <input id="hough-minr" type="range" min={3} max={maxR - 1} value={minR} onChange={e => setMinR(Number(e.target.value))} />
            </div>
            <div className="hough-slider-row">
              <label htmlFor="hough-maxr">Max radius: <strong>{maxR}</strong> px</label>
              <input id="hough-maxr" type="range" min={minR + 1} max={80} value={maxR} onChange={e => setMaxR(Number(e.target.value))} />
            </div>
            <div className="hough-stat">Circles detected: <strong>{circleResult?.circles.length ?? 0}</strong></div>
          </div>
        </div>

        <div className="hough-right">
          <div className="demo-images hough-images">
            <div className="demo-panel">
              <h3>Original</h3>
              <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Original image" />
            </div>
            <div className="demo-panel">
              <h3>Edges (Canny)</h3>
              <PixelCanvas image={edgeImage} size={DISPLAY} label="Canny edge map" />
            </div>
            <div className="demo-panel">
              <h3>Hough Lines</h3>
              <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Detected line segments">
                <svg className="hough-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                  {lineResult?.lines.map((l, i) => (
                    <line key={i} x1={l.x1 + 0.5} y1={l.y1 + 0.5} x2={l.x2 + 0.5} y2={l.y2 + 0.5} className="hough-line" />
                  ))}
                </svg>
              </PixelCanvas>
              <p className="caption">Accumulator: ρ → · θ ↓ (peaks circled)</p>
              <PixelCanvas image={lineSpace} size={DISPLAY} pixelated={false} label="Line accumulator (rho horizontal, theta vertical)">
                {lineResult && (
                  <svg className="hough-overlay" viewBox={`0 0 ${lineResult.numRho} ${lineResult.numTheta}`} preserveAspectRatio="none" aria-hidden="true">
                    {lineResult.lines.map((l, i) => (
                      <ellipse key={i} cx={l.rhoIndex + 0.5} cy={l.thetaIndex + 0.5} rx={8} ry={6} className="hough-peak" />
                    ))}
                  </svg>
                )}
              </PixelCanvas>
            </div>
            <div className="demo-panel">
              <h3>Hough Circles</h3>
              <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Detected circles">
                <svg className="hough-overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
                  {circleResult?.circles.map((c, i) => (
                    <circle key={i} cx={c.x + 0.5} cy={c.y + 0.5} r={c.r} className="hough-circle" />
                  ))}
                </svg>
              </PixelCanvas>
              <p className="caption">Centre accumulator (best score over radii)</p>
              <PixelCanvas image={circleSpace} size={DISPLAY} label="Circle centre accumulator" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
