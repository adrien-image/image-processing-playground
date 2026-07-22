import { useState, useRef, useEffect, useCallback } from 'react';
import { rgbaToGrayscale } from '../processors/utils';
import { CodeViewer } from '../components/CodeViewer';
import templateCode from './TemplateSection.tsx?raw';

const W = 180, H = 180;
const PATCH_W = 30, PATCH_H = 24;

function generateSceneA(): string {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#e94560'; ctx.fillRect(15, 15, 50, 50);
  ctx.fillStyle = '#0f3460'; ctx.beginPath(); ctx.arc(130, 130, 25, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#16c79a'; ctx.fillRect(105, 25, 45, 35);
  ctx.fillStyle = '#f5c518'; ctx.beginPath(); ctx.arc(40, 135, 18, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.fillRect(68, 60, 28, 8);
  ctx.fillRect(79, 60, 8, 28);
  return c.toDataURL();
}

function generateSceneB(): string {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, W, H);
  // Same shapes, DIFFERENT positions
  ctx.fillStyle = '#e94560'; ctx.fillRect(100, 100, 50, 50);
  ctx.fillStyle = '#0f3460'; ctx.beginPath(); ctx.arc(45, 45, 25, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#16c79a'; ctx.fillRect(20, 90, 45, 35);
  ctx.fillStyle = '#f5c518'; ctx.beginPath(); ctx.arc(140, 100, 18, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#aaa'; ctx.fillRect(70, 20, 35, 35);
  // White T shape — moved to (115, 120)
  ctx.fillStyle = '#fff';
  ctx.fillRect(115, 120, 28, 8);
  ctx.fillRect(126, 120, 8, 28);
  return c.toDataURL();
}

function extractPatch(gray: Float64Array, w: number, cx: number, cy: number): Float64Array {
  const hw = Math.floor(PATCH_W / 2), hh = Math.floor(PATCH_H / 2);
  let px = Math.max(0, Math.min(W - PATCH_W, cx - hw));
  let py = Math.max(0, Math.min(H - PATCH_H, cy - hh));
  const patch = new Float64Array(PATCH_W * PATCH_H);
  for (let y = 0; y < PATCH_H; y++)
    for (let x = 0; x < PATCH_W; x++)
      patch[y * PATCH_W + x] = gray[(py + y) * w + (px + x)];
  return patch;
}

function nccMatch(
  image: Float64Array, iw: number, ih: number,
  templ: Float64Array, tw: number, th: number,
): { x: number; y: number; score: number } {
  let tMean = 0;
  for (let i = 0; i < tw * th; i++) tMean += templ[i];
  tMean /= (tw * th);
  let tStd = 0;
  for (let i = 0; i < tw * th; i++) tStd += (templ[i] - tMean) ** 2;
  tStd = Math.sqrt(tStd / (tw * th)) || 1;

  let bestX = 0, bestY = 0, maxScore = -Infinity;
  for (let y = 0; y <= ih - th; y++) {
    for (let x = 0; x <= iw - tw; x++) {
      let iMean = 0;
      for (let ty = 0; ty < th; ty++)
        for (let tx = 0; tx < tw; tx++)
          iMean += image[(y + ty) * iw + (x + tx)];
      iMean /= (tw * th);
      let num = 0, iStd2 = 0;
      for (let ty = 0; ty < th; ty++) {
        for (let tx = 0; tx < tw; tx++) {
          const iv = image[(y + ty) * iw + (x + tx)] - iMean;
          num += iv * (templ[ty * tw + tx] - tMean);
          iStd2 += iv * iv;
        }
      }
      const iStd = Math.sqrt(iStd2 / (tw * th)) || 1;
      const ncc = num / ((tw * th) * tStd * iStd);
      if (ncc > maxScore) { maxScore = ncc; bestX = x; bestY = y; }
    }
  }
  return { x: bestX, y: bestY, score: maxScore };
}

export function TemplateSection() {
  const [patch, setPatch] = useState<Float64Array | null>(null);
  const [match, setMatch] = useState<{ x: number; y: number; score: number } | null>(null);
  const [showCode, setShowCode] = useState(false);
  const sceneARef = useRef<HTMLImageElement>(null);
  const sceneBRef = useRef<HTMLImageElement>(null);
  const templRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef<HTMLCanvasElement>(null);
  const grayARef = useRef<Float64Array | null>(null);
  const grayBRef = useRef<Float64Array | null>(null);

  const runMatch = useCallback((templ: Float64Array) => {
    const grayB = grayBRef.current;
    const imgB = sceneBRef.current;
    if (!grayB || !imgB) return;
    const m = nccMatch(grayB, W, H, templ, PATCH_W, PATCH_H);
    setMatch(m);

    const canvas = resultRef.current!;
    canvas.width = W; canvas.height = H;
    const rCtx = canvas.getContext('2d')!;
    rCtx.drawImage(imgB, 0, 0);
    rCtx.strokeStyle = '#34d399';
    rCtx.lineWidth = 3;
    rCtx.strokeRect(m.x, m.y, PATCH_W, PATCH_H);
    rCtx.fillStyle = 'rgba(52, 211, 153, 0.15)';
    rCtx.fillRect(m.x, m.y, PATCH_W, PATCH_H);
  }, []);

  const handleSceneAClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();
    const cx = Math.round((e.clientX - rect.left) * (W / rect.width));
    const cy = Math.round((e.clientY - rect.top) * (H / rect.height));
    const gray = grayARef.current;
    if (!gray) return;

    const templ = extractPatch(gray, W, cx, cy);
    setPatch(templ);

    // Draw template thumbnail
    const tCanvas = templRef.current;
    if (tCanvas) {
      tCanvas.width = PATCH_W; tCanvas.height = PATCH_H;
      const tCtx = tCanvas.getContext('2d')!;
      const tImg = tCtx.createImageData(PATCH_W, PATCH_H);
      for (let i = 0; i < PATCH_W * PATCH_H; i++) {
        const v = templ[i];
        tImg.data[i * 4] = v; tImg.data[i * 4 + 1] = v;
        tImg.data[i * 4 + 2] = v; tImg.data[i * 4 + 3] = 255;
      }
      tCtx.putImageData(tImg, 0, 0);
    }

    runMatch(templ);
  }, [runMatch]);

  // Load both scenes
  useEffect(() => {
    const imgA = sceneARef.current;
    const imgB = sceneBRef.current;
    if (!imgA || !imgB) return;
    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded < 2) return;
      const toGray = (img: HTMLImageElement) => {
        const c = document.createElement('canvas');
        c.width = W; c.height = H;
        const ctx = c.getContext('2d')!;
        ctx.drawImage(img, 0, 0);
        return rgbaToGrayscale(ctx.getImageData(0, 0, W, H).data, W, H);
      };
      grayARef.current = toGray(imgA);
      grayBRef.current = toGray(imgB);
    };
    imgA.onload = onLoad; imgB.onload = onLoad;
    imgA.src = generateSceneA(); imgB.src = generateSceneB();
    if (imgA.complete && imgA.naturalWidth && imgB.complete && imgB.naturalWidth) onLoad();
    return () => { imgA.onload = imgB.onload = null; };
  }, []);

  const disp = 260;

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Template Matching — NCC — Click Scene A to pick a template</span>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Scene A — pick template</h3>
          <img ref={sceneARef} className="color-img" alt="scene A"
            onClick={handleSceneAClick}
            style={{ width: disp, height: disp, imageRendering: 'pixelated', cursor: 'crosshair' }} />
        </div>
        <div className="demo-panel">
          <h3>Template ({PATCH_W}×{PATCH_H})</h3>
          <canvas ref={templRef} className="color-img"
            style={{ width: PATCH_W * 6, height: PATCH_H * 6, imageRendering: 'pixelated' }} />
        </div>
        <div className="demo-panel">
          <h3>Scene B — find match</h3>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <img ref={sceneBRef} className="color-img" alt="scene B"
              style={{ width: disp, height: disp, imageRendering: 'pixelated' }} />
            <canvas ref={resultRef} className="color-img" alt="result"
              style={{ position: 'absolute', top: 0, left: 0, width: disp, height: disp, imageRendering: 'pixelated', pointerEvents: 'none' }} />
          </div>
        </div>
      </div>

      {match && (
        <div className="demo-slider-section">
          <div className="demo-slider-row" style={{ justifyContent: 'center', gap: '2rem' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              Match at <strong style={{ color: 'var(--success)' }}>({match.x}, {match.y})</strong>
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
              NCC score: <strong style={{ color: match.score > 0.5 ? 'var(--success)' : 'var(--error)' }}>{match.score.toFixed(3)}</strong>
            </span>
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Template Matching — Normalised Cross-Correlation</h4>
        <p><strong>Click on Scene A</strong> to extract a {PATCH_W}×{PATCH_H} template.
        The algorithm then searches for it in <strong>Scene B</strong> — a different image where the shapes are at new positions.</p>
        <p>The <span style={{ color: '#34d399' }}>green rectangle</span> on Scene B shows where NCC found the best match.
        Since the scenes use the same shapes but rearranged, the match should correctly locate the template in its new position.</p>
        <p><strong>NCC</strong> is robust to global brightness changes — it normalises both the template and each patch,
        so a shape that's slightly darker or lighter in Scene B still matches correctly.</p>
      </div>
    </div>
    {showCode && <CodeViewer code={templateCode} title="TemplateSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
