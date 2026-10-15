import { useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react';
import nccCode from './ncc.ts?raw';
import pythonCode from './ncc.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { grayToImageData, rgbImage } from '../../lib/image';
import { hotColor } from '../hough/hough';
import { extractPatch, nccMatch } from './ncc';
import './template.css';

const W = 180, H = 180;
const PATCH_W = 30, PATCH_H = 24;
const DISPLAY = 260;

function sceneA(): HTMLCanvasElement {
  return makeCanvas(W, H, ctx => {
    ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#e94560'; ctx.fillRect(15, 15, 50, 50);
    ctx.fillStyle = '#0f3460'; ctx.beginPath(); ctx.arc(130, 130, 25, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#16c79a'; ctx.fillRect(105, 25, 45, 35);
    ctx.fillStyle = '#f5c518'; ctx.beginPath(); ctx.arc(40, 135, 18, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillRect(68, 60, 28, 8);
    ctx.fillRect(79, 60, 8, 28);
  });
}

function sceneB(): HTMLCanvasElement {
  return makeCanvas(W, H, ctx => {
    ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, W, H);
    // Same shapes, different positions
    ctx.fillStyle = '#e94560'; ctx.fillRect(100, 100, 50, 50);
    ctx.fillStyle = '#0f3460'; ctx.beginPath(); ctx.arc(45, 45, 25, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#16c79a'; ctx.fillRect(20, 90, 45, 35);
    ctx.fillStyle = '#f5c518'; ctx.beginPath(); ctx.arc(140, 100, 18, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#aaa'; ctx.fillRect(70, 20, 35, 35);
    ctx.fillStyle = '#fff';
    ctx.fillRect(115, 120, 28, 8);
    ctx.fillRect(126, 120, 8, 28);
  });
}

/** Top-left corner of the patch centred on (cx, cy), kept inside the image. */
const patchCorner = (cx: number, cy: number) => ({
  x: Math.max(0, Math.min(W - PATCH_W, Math.round(cx - PATCH_W / 2))),
  y: Math.max(0, Math.min(H - PATCH_H, Math.round(cy - PATCH_H / 2))),
});

const pct = (v: number, of: number) => `${(v / of) * 100}%`;
const box = (x: number, y: number) => ({ left: pct(x, W), top: pct(y, H), width: pct(PATCH_W, W), height: pct(PATCH_H, H) });

export function TemplateSection() {
  const [showCode, setShowCode] = useState(false);
  const [corner, setCorner] = useState<{ x: number; y: number } | null>(null);
  // Keyboard cursor (patch corner) before Enter confirms it.
  const [cursor, setCursor] = useState(() => patchCorner(W / 2, H / 2));
  const canvasA = useMemo(sceneA, []);
  const canvasB = useMemo(sceneB, []);
  const { image: imgA } = useImageData(canvasA, W);
  const { image: imgB } = useImageData(canvasB, W);

  const result = useMemo(() => {
    if (!imgA || !imgB || !corner) return null;
    const templ = extractPatch(imgA.gray, W, corner.x, corner.y, PATCH_W, PATCH_H);
    return { templ, match: nccMatch(imgB.gray, W, H, templ, PATCH_W, PATCH_H) };
  }, [imgA, imgB, corner]);

  const templateImage = useMemo(() => (result ? grayToImageData(result.templ, PATCH_W, PATCH_H) : null), [result]);
  const scoreImage = useMemo(() => {
    const m = result?.match;
    if (!m) return null;
    // Map NCC from [−1, 1] to the hot colormap.
    return rgbImage(m.scoresWidth, m.scoresHeight, i => hotColor((m.scores[i] + 1) / 2));
  }, [result]);

  const pick = (e: MouseEvent<HTMLDivElement>) => {
    // Measure the image box itself, not the (possibly wider) focusable wrapper.
    const imageBox = e.currentTarget.querySelector('.pixel-canvas') ?? e.currentTarget;
    const rect = imageBox.getBoundingClientRect();
    const c = patchCorner(((e.clientX - rect.left) / rect.width) * W, ((e.clientY - rect.top) / rect.height) * H);
    setCursor(c);
    setCorner(c);
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 2;
    const moves: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (moves[e.key]) {
      e.preventDefault();
      const [dx, dy] = moves[e.key];
      setCursor(c => patchCorner(c.x + PATCH_W / 2 + dx, c.y + PATCH_H / 2 + dy));
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setCorner(cursor);
    }
  };

  const match = result?.match ?? null;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Template Matching (NCC) — click Scene A, or focus it and use the arrow keys + Enter, to pick a template</span>
        <span className="ml-auto">
          <ShowCodeButton code={nccCode} title="template/ncc.ts" python={{ code: pythonCode, title: 'template/ncc.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Scene A — pick template</h3>
          <div
            className="template-picker"
            role="button"
            tabIndex={0}
            aria-label={`Scene A. Template at ${cursor.x}, ${cursor.y}. Arrow keys move, Enter selects.`}
            onClick={pick}
            onKeyDown={onKey}
          >
            <PixelCanvas image={imgA?.imageData ?? null} size={DISPLAY} label="Scene A">
              <div className="template-box template-box-cursor" style={box(cursor.x, cursor.y)} />
              {corner && <div className="template-box template-box-selected" style={box(corner.x, corner.y)} />}
            </PixelCanvas>
          </div>
        </div>
        <div className="demo-panel">
          <h3>Template ({PATCH_W}×{PATCH_H})</h3>
          <PixelCanvas image={templateImage} size={180} placeholderAspect={PATCH_W / PATCH_H} label="Selected template" placeholder="Click Scene A" />
        </div>
        <div className="demo-panel">
          <h3>Scene B — find match</h3>
          <PixelCanvas image={imgB?.imageData ?? null} size={DISPLAY} label="Scene B">
            {match && <div className="template-box template-box-match" style={box(match.x, match.y)} />}
          </PixelCanvas>
        </div>
        <div className="demo-panel">
          <h3>NCC score map</h3>
          <PixelCanvas
            image={scoreImage}
            size={200}
            placeholderAspect={(W - PATCH_W + 1) / (H - PATCH_H + 1)}
            label="NCC score at every template position"
            placeholder={result ? 'No score map for a flat patch' : 'Pick a template'}
          >
            {match && <div className="template-peak" style={{ left: pct(match.x + 0.5, match.scoresWidth), top: pct(match.y + 0.5, match.scoresHeight) }} />}
          </PixelCanvas>
          <p className="caption">Black = −1 · white = +1 (best match ringed)</p>
        </div>
      </div>

      {result && (
        <div className="demo-slider-section">
          {match ? (
            <div className="demo-slider-row template-result">
              <span className="mono">Match at <strong className="text-success">({match.x}, {match.y})</strong></span>
              <span className="mono">NCC score: <strong className={match.score > 0.5 ? 'text-success' : 'text-error'}>{match.score.toFixed(3)}</strong></span>
            </div>
          ) : (
            <p className="note-warning" role="alert">
              ⚠ That patch is flat — it has no texture, so NCC is undefined (it would match every flat area equally).
              Pick a patch containing an edge or corner.
            </p>
          )}
        </div>
      )}

      <div className="demo-explanation">
        <h4>Template Matching — Normalised Cross-Correlation</h4>
        <p><strong>Pick a patch in Scene A</strong> to extract a {PATCH_W}×{PATCH_H} template.
        The algorithm slides it over every position in <strong>Scene B</strong> — a different image where the shapes are at new positions — and scores each one.</p>
        <p>The <span className="text-success">green rectangle</span> marks the best score; the score map shows how strongly every position matched.
        NCC subtracts the mean and divides by the standard deviation of both the template and the image window, so a shape that is uniformly darker or lighter in Scene B still scores ~1.</p>
        <p>Because of that normalisation, a perfectly flat patch can't be matched: its standard deviation is zero.</p>
      </div>
    </div>
  );
}
