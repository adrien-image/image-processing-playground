import { CodeViewer } from '../components/CodeViewer';
import houghCode from './HoughSection.tsx?raw';
import { useState, useEffect, useRef, useCallback } from 'react';
import { rgbaToGrayscale } from '../processors/utils';
import { convolve } from '../processors/utils';

function generateTestImage(): string {
  const c = document.createElement('canvas');
  c.width = 200; c.height = 200;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, 200, 200);
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
  // Lines
  ctx.beginPath(); ctx.moveTo(30, 30); ctx.lineTo(160, 120); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(40, 140); ctx.lineTo(170, 60); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(90, 20); ctx.lineTo(80, 170); ctx.stroke();
  ctx.strokeRect(40, 40, 120, 120);
  // Circles
  ctx.beginPath(); ctx.arc(140, 140, 30, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(75, 85, 18, 0, Math.PI * 2); ctx.stroke();
  return c.toDataURL();
}

/** Hot colormap: 0=black → red → orange → yellow → white=1 */
function hotColor(t: number): [number, number, number] {
  const r = Math.min(1, t * 2);
  const g = Math.min(1, Math.max(0, t * 2 - 0.5) * 2);
  const b = Math.min(1, Math.max(0, t * 2 - 1) * 2);
  return [r * 255, g * 255, b * 255];
}

import { SOBEL_X, SOBEL_Y } from '../processors/sobel/kernels';

interface DetectedLine { rho: number; theta: number; votes: number; x1: number; y1: number; x2: number; y2: number; }
interface DetectedCircle { x: number; y: number; r: number; votes: number; }

function clipLine(rho: number, theta: number, w: number, h: number): { x1: number; y1: number; x2: number; y2: number } | null {
  const cos = Math.cos(theta), sin = Math.sin(theta);
  const pts: number[][] = [];
  const edges = [[0, 0, w, 0], [w, 0, w, h], [w, h, 0, h], [0, h, 0, 0]];
  for (const [x1, y1, x2, y2] of edges) {
    const denom = (x1 - x2) * cos + (y1 - y2) * sin;
    if (Math.abs(denom) < 1e-10) continue;
    const t = ((x1 * cos + y1 * sin) - rho) / denom;
    if (t >= 0 && t <= 1) {
      const px = x1 + t * (x2 - x1), py = y1 + t * (y2 - y1);
      if (px >= -1 && px <= w + 1 && py >= -1 && py <= h + 1) pts.push([px, py]);
    }
  }
  if (pts.length < 2) return null;
  return { x1: pts[0][0], y1: pts[0][1], x2: pts[1][0], y2: pts[1][1] };
}

function detectLines(edgePixels: Float64Array, w: number, h: number, threshold: number, minLength: number):
  { lines: DetectedLine[]; accumulator: Uint32Array; numTheta: number; numRho: number; maxRho: number } {
  const maxRho = Math.sqrt(w * w + h * h);
  const numTheta = 180;
  const numRho = Math.round(maxRho * 2);
  const acc = new Uint32Array(numTheta * numRho);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (edgePixels[y * w + x] > 128)
        for (let ti = 0; ti < numTheta; ti++) {
          const theta = (ti / numTheta) * Math.PI;
          const rho = x * Math.cos(theta) + y * Math.sin(theta);
          const ri = Math.round((rho + maxRho) / (2 * maxRho) * (numRho - 1));
          if (ri >= 0 && ri < numRho) acc[ti * numRho + ri]++;
        }
  const lines: DetectedLine[] = [];
  for (let ti = 0; ti < numTheta; ti++)
    for (let ri = 0; ri < numRho; ri++) {
      const votes = acc[ti * numRho + ri];
      if (votes < threshold) continue;
      let isMax = true;
      for (let dt = -1; dt <= 1 && isMax; dt++)
        for (let dr = -1; dr <= 1 && isMax; dr++) {
          const nt = ti + dt, nr = ri + dr;
          if (nt >= 0 && nt < numTheta && nr >= 0 && nr < numRho && (dt !== 0 || dr !== 0))
            if (acc[nt * numRho + nr] >= votes) isMax = false;
        }
      if (!isMax) continue;
      const theta = (ti / numTheta) * Math.PI;
      const rho = -maxRho + (ri / (numRho - 1)) * 2 * maxRho;
      const clipped = clipLine(rho, theta, w, h);
      if (clipped) {
        const dx = clipped.x2 - clipped.x1, dy = clipped.y2 - clipped.y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len >= minLength) lines.push({ rho, theta, votes, ...clipped });
      }
    }
  lines.sort((a, b) => b.votes - a.votes);
  return { lines: lines.slice(0, 20), accumulator: acc, numTheta, numRho, maxRho };
}

function detectCircles(edgePixels: Float64Array, w: number, h: number, minR: number, maxR: number, threshold: number):
  { circles: DetectedCircle[]; accumulator: Uint32Array; accW: number; accH: number } {
  const circles: DetectedCircle[] = [];
  const accSum = new Uint32Array(w * h);
  const acc = new Uint32Array(w * h); // reusable buffer
  for (let r = minR; r <= maxR; r++) {
    acc.fill(0);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (edgePixels[y * w + x] > 128)
          for (let a = 0; a < 360; a += 10) {
            const angle = (a * Math.PI) / 180;
            const cx = Math.round(x - r * Math.cos(angle));
            const cy = Math.round(y - r * Math.sin(angle));
            if (cx >= 0 && cx < w && cy >= 0 && cy < h) acc[cy * w + cx]++;
          }
    for (let cy = 0; cy < h; cy++)
      for (let cx = 0; cx < w; cx++) {
        accSum[cy * w + cx] += acc[cy * w + cx];
        const votes = acc[cy * w + cx];
        if (votes < threshold) continue;
        let isMax = true;
        for (let dy = -2; dy <= 2 && isMax; dy++)
          for (let dx = -2; dx <= 2 && isMax; dx++) {
            const nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h && (dx !== 0 || dy !== 0))
              if (acc[ny * w + nx] >= votes) isMax = false;
          }
        if (isMax) circles.push({ x: cx, y: cy, r, votes });
      }
  }
  circles.sort((a, b) => b.votes - a.votes);
  return { circles: circles.slice(0, 10), accumulator: accSum, accW: w, accH: h };
}

/** Shared edge detection */
function computeEdges(img: HTMLImageElement): { edges: Float64Array; w: number; h: number } {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, w, h);
  const gray = rgbaToGrayscale(imageData.data, w, h);
  const gx = convolve(gray, w, h, SOBEL_X);
  const gy = convolve(gray, w, h, SOBEL_Y);
  const edges = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) edges[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
  let maxE = 0;
  for (let i = 0; i < w * h; i++) if (edges[i] > maxE) maxE = edges[i];
  const s = maxE > 0 ? 255 / maxE : 1;
  for (let i = 0; i < w * h; i++) edges[i] = Math.min(255, edges[i] * s);
  return { edges, w, h };
}

function drawEdges(canvas: HTMLCanvasElement, edges: Float64Array, w: number, h: number) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = edges[i];
    rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
}

export function HoughSection() {
  const srcRef = useRef<HTMLImageElement>(null);
  const edgeRef = useRef<HTMLCanvasElement>(null);
  const lineRef = useRef<HTMLCanvasElement>(null);
  const lineSpaceRef = useRef<HTMLCanvasElement>(null);
  const circleRef = useRef<HTMLCanvasElement>(null);
  const circleSpaceRef = useRef<HTMLCanvasElement>(null);
  const [lineThresh, setLineThresh] = useState(50);
  const [showCode, setShowCode] = useState(false);
  const [minLen, setMinLen] = useState(20);
  const [circThresh, setCircThresh] = useState(25);
  const [minR, setMinR] = useState(8);
  const [maxR, setMaxR] = useState(35);
  const [lineCount, setLineCount] = useState(0);
  const [circCount, setCircCount] = useState(0);

  const process = useCallback(() => {
    const img = srcRef.current;
    if (!img || !img.complete) return;
    const { edges, w, h } = computeEdges(img);

    // Draw edges
    if (edgeRef.current) drawEdges(edgeRef.current, edges, w, h);

    // Lines
    const { lines, accumulator: lineAcc, numTheta, numRho } = detectLines(edges, w, h, lineThresh, minLen);
    setLineCount(lines.length);
    const lCanv = lineRef.current;
    if (lCanv) {
      lCanv.width = w; lCanv.height = h;
      const ctx = lCanv.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 2;
      for (const line of lines) {
        ctx.beginPath(); ctx.moveTo(line.x1, line.y1); ctx.lineTo(line.x2, line.y2); ctx.stroke();
      }
    }
    // Line Hough space with heat colormap
    const lsCanv = lineSpaceRef.current;
    if (lsCanv) {
      lsCanv.width = numTheta; lsCanv.height = numRho;
      const lsCtx = lsCanv.getContext('2d')!;
      let maxV = 0;
      for (let i = 0; i < lineAcc.length; i++) if (lineAcc[i] > maxV) maxV = lineAcc[i];
      const imageData = lsCtx.createImageData(numTheta, numRho);
      for (let ti = 0; ti < numTheta; ti++)
        for (let ri = 0; ri < numRho; ri++) {
          const t = maxV > 0 ? lineAcc[ti * numRho + ri] / maxV : 0;
          const idx = (ri * numTheta + ti) * 4;
          // Hot colormap
          const [r, g, b] = hotColor(t);
          imageData.data[idx] = r;
          imageData.data[idx + 1] = g * 255;
          imageData.data[idx + 2] = b * 255;
          imageData.data[idx + 3] = 255;
        }
      lsCtx.putImageData(imageData, 0, 0);
    }

    // Circles
    const { circles, accumulator: circAcc, accW, accH } = detectCircles(edges, w, h, minR, maxR, circThresh);
    setCircCount(circles.length);
    const cCanv = circleRef.current;
    if (cCanv) {
      cCanv.width = w; cCanv.height = h;
      const ctx = cCanv.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      ctx.strokeStyle = '#44ff44'; ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(68, 255, 68, 0.12)';
      for (const circ of circles) {
        ctx.beginPath(); ctx.arc(circ.x, circ.y, circ.r, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(circ.x, circ.y, circ.r, 0, Math.PI * 2); ctx.stroke();
      }
    }
    // Circle Hough space with heat colormap
    const csCanv = circleSpaceRef.current;
    if (csCanv && accW > 0) {
      csCanv.width = accW; csCanv.height = accH;
      const csCtx = csCanv.getContext('2d')!;
      let maxV = 0;
      for (let i = 0; i < circAcc.length; i++) if (circAcc[i] > maxV) maxV = circAcc[i];
      const imageData = csCtx.createImageData(accW, accH);
      for (let i = 0; i < accW * accH; i++) {
        const t = maxV > 0 ? circAcc[i] / maxV : 0;
        const [r, g, b] = hotColor(t);
        imageData.data[i * 4] = r;
        imageData.data[i * 4 + 1] = g;
        imageData.data[i * 4 + 2] = b;
        imageData.data[i * 4 + 3] = 255;
      }
      csCtx.putImageData(imageData, 0, 0);
    }
  }, [lineThresh, minLen, circThresh, minR, maxR]);

  useEffect(() => {
    const img = srcRef.current;
    if (!img) return;
    const onLoad = () => process();
    img.onload = onLoad;
    img.src = generateTestImage();
    if (img.complete && img.naturalWidth) process();
    return () => { img.onload = null; };
  }, [process]);

  return (
    <>
    <div className="demo-section">
      <div className="hough-layout">
        {/* Left: Explanation panel */}
        <div className="hough-left">
          <div className="demo-explanation" style={{ boxShadow: 'none', border: 'none', background: 'transparent', padding: 0 }}>
            <h3>Hough Line Transform</h3>
            <p>Each edge pixel votes for all possible lines <strong>(ρ, θ)</strong> passing through it.
            Peaks in the accumulator reveal the most likely lines.</p>
            <div className="demo-slider-section" style={{ marginTop: '0.5rem', padding: '0.5rem' }}>
              <div className="hough-slider-row">
                <span>Vote threshold: <strong>{lineThresh}</strong></span>
                <input type="range" min={10} max={200} value={lineThresh}
                  onChange={e => setLineThresh(Number(e.target.value))} />
              </div>
              <div className="hough-slider-row">
                <span>Min length: <strong>{minLen}</strong>px</span>
                <input type="range" min={5} max={180} value={minLen}
                  onChange={e => setMinLen(Number(e.target.value))} />
              </div>
              <div className="hough-stat">Lines detected: <strong>{lineCount}</strong></div>
            </div>

            <h3 style={{ marginTop: '1rem' }}>Hough Circle Transform</h3>
            <p>Each edge pixel votes for possible circle centres <strong>(a, b)</strong> at each radius.
            Adjust the radius range and vote threshold to tune detection.</p>
            <div className="demo-slider-section" style={{ marginTop: '0.5rem', padding: '0.5rem' }}>
              <div className="hough-slider-row">
                <span>Vote threshold: <strong>{circThresh}</strong></span>
                <input type="range" min={5} max={100} value={circThresh}
                  onChange={e => setCircThresh(Number(e.target.value))} />
              </div>
              <div className="hough-slider-row">
                <span>Min radius: <strong>{minR}</strong>px</span>
                <input type="range" min={3} max={Math.max(3, maxR - 1)} value={minR}
                  onChange={e => setMinR(Number(e.target.value))} />
              </div>
              <div className="hough-slider-row">
                <span>Max radius: <strong>{maxR}</strong>px</span>
                <input type="range" min={minR + 1} max={60} value={maxR}
                  onChange={e => setMaxR(Number(e.target.value))} />
              </div>
              <div className="hough-stat">Circles detected: <strong>{circCount}</strong></div>
              <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginTop: '0.5rem' }}>💻 Show code</button>
            </div>
          </div>
        </div>

        {/* Right: Images */}
        <div className="hough-right">
          <div className="demo-images" style={{ justifyContent: 'flex-start' }}>
            <div className="demo-panel" style={{ flex: 'none' }}>
              <h3>Original</h3>
              <img ref={srcRef} className="color-img" alt="original" style={{ width: 200, height: 200 }} />
            </div>
            <div className="demo-panel" style={{ flex: 'none' }}>
              <h3>Edges (Sobel)</h3>
              <canvas ref={edgeRef} className="color-img" style={{ width: 200, height: 200 }} />
            </div>
            <div className="demo-panel" style={{ flex: 'none' }}>
              <h3>Hough Lines</h3>
              <canvas ref={lineRef} className="color-img" style={{ width: 200, height: 200 }} />
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>θ → (horizontal) · ρ → (vertical)</div>
              <canvas ref={lineSpaceRef} style={{ width: 180, height: 80, marginTop: '0.3rem', borderRadius: 'var(--radius-sm)', background: '#000' }} />
            </div>
            <div className="demo-panel" style={{ flex: 'none' }}>
              <h3>Hough Circles</h3>
              <canvas ref={circleRef} className="color-img" style={{ width: 200, height: 200 }} />
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>Centre accumulator (summed over radii)</div>
              <canvas ref={circleSpaceRef} style={{ width: 200, height: 80, marginTop: '0.3rem', borderRadius: 'var(--radius-sm)', background: '#000' }} />
            </div>
          </div>
        </div>
      </div>
    </div>
      {showCode && <CodeViewer code={houghCode} title="HoughSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
