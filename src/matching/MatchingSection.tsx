import { CodeViewer } from '../components/CodeViewer';
import harrisCode from './harris.ts?raw';
import { useState, useCallback, useRef, useEffect } from 'react';
import { rgbaToGrayscale } from '../processors/utils';
import { detectHarris } from './harris';
import { matchKeypoints, matchPoints } from './descriptors';
import { estimateTransform, formatTransform } from './ransac';
import type { SimTransformResult } from './ransac';
import type { Keypoint } from './harris';
import type { Match } from './descriptors';

interface CachedImage {
  pixels: Float64Array;
  w: number;
  h: number;
  img: HTMLImageElement;
}

// The known ground-truth transform applied to Image B (for comparison)
const GT_ANGLE = 0.4;
const GT_SCALE = 0.85;
const GT_TX = 8;
const GT_TY = -6;
const GT_DEG = (GT_ANGLE * 180 / Math.PI).toFixed(1);

// Effective translation after center-of-rotation offset.
// Canvas applies: translate(+center+GT) → rotate → scale → translate(−center).
// Expanding into x' = s·R·x + t gives:
const GT_C = Math.cos(GT_ANGLE);
const GT_S = Math.sin(GT_ANGLE);
const W = 220, H = 200;
const GT_TX_EFF = W / 2 * (1 - GT_SCALE * GT_C) + GT_SCALE * GT_S * H / 2 + GT_TX;
const GT_TY_EFF = H / 2 * (1 - GT_SCALE * GT_C) - GT_SCALE * GT_S * W / 2 + GT_TY;

function generateShapeImage(width: number, height: number, transform?: { rotate: number; scale: number; tx: number; ty: number }): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  if (transform) {
    ctx.translate(width / 2 + transform.tx, height / 2 + transform.ty);
    ctx.rotate(transform.rotate);
    ctx.scale(transform.scale, transform.scale);
    ctx.translate(-width / 2, -height / 2);
  }
  ctx.fillStyle = '#ddd';
  ctx.fillRect(40, 40, 120, 80);
  ctx.fillStyle = '#888';
  ctx.fillRect(60, 60, 60, 60);
  ctx.fillStyle = '#aaa';
  ctx.beginPath(); ctx.arc(180, 60, 30, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#666';
  ctx.beginPath(); ctx.arc(180, 120, 20, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.strokeRect(30, 150, 140, 50);
  ctx.fillStyle = '#999'; ctx.fillRect(50, 160, 30, 30);
  ctx.restore();
  return canvas.toDataURL();
}

export function MatchingSection() {
  const imgARef = useRef<HTMLImageElement>(null);
  const imgBRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cachedA = useRef<CachedImage | null>(null);
  const cachedB = useRef<CachedImage | null>(null);
  const [status, setStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle');
  const [numMatches, setNumMatches] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transformText, setTransformText] = useState('');

  useEffect(() => {
    const w = 220, h = 200;
    const imgA = generateShapeImage(w, h);
    const imgB = generateShapeImage(w, h, { rotate: 0.4, scale: 0.85, tx: 8, ty: -6 });

    // Load images and pre-compute grayscale data for fast re-runs
    const loadAndCache = async () => {
      const readImg = (src: string): Promise<HTMLImageElement> =>
        new Promise((resolve) => { const i = new Image(); i.onload = () => resolve(i); i.src = src; });

      const imgAEl = await readImg(imgA);
      const imgBEl = await readImg(imgB);

      if (imgARef.current) imgARef.current.src = imgA;
      if (imgBRef.current) imgBRef.current.src = imgB;

      const toGray = (img: HTMLImageElement): CachedImage => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        return {
          pixels: rgbaToGrayscale(data.data, canvas.width, canvas.height),
          w: canvas.width,
          h: canvas.height,
          img,
        };
      };

      cachedA.current = toGray(imgAEl);
      cachedB.current = toGray(imgBEl);
    };

    loadAndCache();
  }, []);

  const findMatches = useCallback(async () => {
    setStatus('processing');
    setError(null);

    try {
      // Yield to let React render the "processing" button state
      await new Promise(r => setTimeout(r, 0));

      // Use cached grayscale data (avoids re-reading canvases)
      const imgA = cachedA.current;
      const imgB = cachedB.current;
      if (!imgA || !imgB) return;

      // Detect Harris corners
      const kpsA: Keypoint[] = detectHarris(imgA.pixels, imgA.w, imgA.h, 500, 0.06);
      const kpsB: Keypoint[] = detectHarris(imgB.pixels, imgB.w, imgB.h, 500, 0.06);

      // Match keypoints
      const matches: Match[] = matchKeypoints(
        imgA.pixels, imgA.w, imgA.h, kpsA,
        imgB.pixels, imgB.w, imgB.h, kpsB,
      );

      if (matches.length < 2) {
        setStatus('done');
        setNumMatches(0);
        setTransformText('Not enough matches found');
        return;
      }

      // Compute point pairs once — reuse for RANSAC and canvas drawing
      const pairs = matches.map(m => matchPoints(m, kpsA, kpsB));

      // Least-squares similarity transform estimation
      const srcPoints = pairs.map(p => p[0]);
      const dstPoints = pairs.map(p => p[1]);
      const result: SimTransformResult = estimateTransform(srcPoints, dstPoints);

      // Draw matches on canvas
      const canvas = canvasRef.current!;
      canvas.width = imgA.w + imgB.w;
      canvas.height = Math.max(imgA.h, imgB.h);
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw images
      ctx.drawImage(imgA.img, 0, 0);
      ctx.drawImage(imgB.img, imgA.w, 0);

      // Draw all matches
      for (let i = 0; i < pairs.length; i++) {
        const [pA, pB] = pairs[i];
        ctx.strokeStyle = 'rgba(52, 211, 153, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pA.x, pA.y);
        ctx.lineTo(pB.x + imgA.w, pB.y);
        ctx.stroke();

        ctx.fillStyle = '#34d399';
        ctx.beginPath(); ctx.arc(pA.x, pA.y, 2, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(pB.x + imgA.w, pB.y, 2, 0, Math.PI * 2); ctx.fill();
      }

      setNumMatches(matches.length);
      setTransformText(formatTransform(result));
      setStatus('done');

    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Matching failed');
    }
  }, []);

  return (
    <>
    <div className="matching-section">
      <div className="matching-topbar">
        <span className="matching-status">
          {status === 'idle' && 'Click "Find Matches" to run Harris corner detector + least-squares similarity estimation.'}
          {status === 'processing' && '⏳ Detecting corners, matching, estimating transform…'}
          {status === 'done' && numMatches > 0 && `✅ ${numMatches} matches found`}
          {status === 'done' && numMatches === 0 && '⚠ Not enough matches found — try a different image pair.'}
          {status === 'error' && `❌ ${error}`}
        </span>
        <button className="btn btn-primary" onClick={findMatches} disabled={status === 'processing'}>
          {status === 'idle' && '🔍 Find Matches'}
          {status === 'done' && '🔄 Re-run'}
          {status === 'processing' && '⏳ Working…'}
        </button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      <div className="matching-images">
        <div className="matching-panel">
          <h3>Image A (original)</h3>
          <img ref={imgARef} className="matching-img" alt="Image A" />
        </div>
        <div className="matching-panel">
          <h3>Image B (transformed)</h3>
          <img ref={imgBRef} className="matching-img" alt="Image B" />
        </div>
        <div className="matching-panel">
          <h3>Matches</h3>
          <canvas ref={canvasRef} className="matching-canvas-wide" />
        </div>
      </div>

      {status === 'done' && numMatches > 0 && (
        <div className="matching-results">
          <div className="matching-result-grid">
            <div className="matching-stat">
              <span className="matching-stat-label">Total matches</span>
              <span className="matching-stat-value">{numMatches}</span>
            </div>
            <div className="matching-stat">
              <span className="matching-stat-label">Transform model</span>
              <span className="matching-stat-value">Similarity</span>
            </div>
          </div>
          <div className="matching-transform">
            <h4>Transform</h4>
            <div className="matching-transform-row">
              <div className="matching-transform-col">
                <span className="matching-transform-label">Estimated</span>
                <pre className="matching-matrix">{transformText}</pre>
              </div>
              <div className="matching-transform-col">
                <span className="matching-transform-label">Ground truth</span>
                <pre className="matching-matrix" style={{ color: 'var(--text-muted)' }}>
θ = {GT_DEG}°<br />tx = {GT_TX_EFF.toFixed(1)}<br />ty = {GT_TY_EFF.toFixed(1)}</pre>
              </div>
            </div>
            <p className="matching-transform-desc">
              Least-squares estimate using all {numMatches} matched point pairs.
            </p>
          </div>
        </div>
      )}
    </div>
      {showCode && <CodeViewer code={harrisCode} title="harris.ts" onClose={() => setShowCode(false)} />}
    </>
  );
}
