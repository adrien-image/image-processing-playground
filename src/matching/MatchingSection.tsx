import { useMemo, useState } from 'react';
import harrisCode from './harris.ts?raw';
import descriptorsCode from './descriptors.ts?raw';
import transformCode from './transform.ts?raw';
import { CodeViewer } from '../components/CodeViewer';
import { PixelCanvas } from '../components/PixelCanvas';
import { useImageData } from '../hooks/useImageData';
import { createCanvas } from '../lib/canvas';
import { detectHarris, type Keypoint } from './harris';
import { matchKeypoints } from './descriptors';
import { formatSimilarity, ransacSimilarity, type Similarity } from './transform';
import { DEFAULT_TRANSFORM, SCENE_HEIGHT, SCENE_WIDTH, drawScene, groundTruth, randomTransform, type SceneTransform } from './scene';
import './matching.css';

interface Pair {
  a: Keypoint;
  b: Keypoint;
  inlier: boolean;
}

type Outcome =
  | { kind: 'ok'; pairs: Pair[]; transform: Similarity; inliers: number; kpsA: number; kpsB: number }
  | { kind: 'too-few'; matches: number; kpsA: number; kpsB: number };

const CODE = {
  harris: { title: 'matching/harris.ts', code: harrisCode },
  descriptors: { title: 'matching/descriptors.ts', code: descriptorsCode },
  transform: { title: 'matching/transform.ts', code: transformCode },
} as const;

export function MatchingSection() {
  const [sceneT, setSceneT] = useState<SceneTransform>(DEFAULT_TRANSFORM);
  const [showCode, setShowCode] = useState<keyof typeof CODE | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canvasA = useMemo(() => drawScene(), []);
  const canvasB = useMemo(() => drawScene(sceneT), [sceneT]);
  const { image: imgA } = useImageData(canvasA, 1024);
  const { image: imgB } = useImageData(canvasB, 1024);
  const gt = groundTruth(sceneT);

  // Both images side by side, for the match lines overlay.
  const sideBySide = useMemo(() => {
    if (!imgA || !imgB) return null;
    const { canvas, ctx } = createCanvas(imgA.width + imgB.width, Math.max(imgA.height, imgB.height));
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(imgA.imageData, 0, 0);
    ctx.putImageData(imgB.imageData, imgA.width, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }, [imgA, imgB]);

  const findMatches = () => {
    if (!imgA || !imgB) return;
    setError(null);
    try {
      const kpsA = detectHarris(imgA.gray, imgA.width, imgA.height, 500, 0.06);
      const kpsB = detectHarris(imgB.gray, imgB.width, imgB.height, 500, 0.06);
      const matches = matchKeypoints(imgA.gray, imgA.width, imgA.height, kpsA, imgB.gray, imgB.width, imgB.height, kpsB);
      const src = matches.map(m => kpsA[m.srcIdx]);
      const dst = matches.map(m => kpsB[m.dstIdx]);
      const ransac = ransacSimilarity(src, dst);
      if (!ransac) {
        setOutcome({ kind: 'too-few', matches: matches.length, kpsA: kpsA.length, kpsB: kpsB.length });
        return;
      }
      setOutcome({
        kind: 'ok',
        pairs: src.map((a, i) => ({ a, b: dst[i], inlier: ransac.inliers[i] })),
        transform: ransac.transform,
        inliers: ransac.inlierCount,
        kpsA: kpsA.length,
        kpsB: kpsB.length,
      });
    } catch (err) {
      setOutcome(null);
      setError(err instanceof Error ? err.message : 'Matching failed');
    }
  };

  const newTransform = () => {
    setSceneT(randomTransform());
    setOutcome(null);
  };

  const ready = Boolean(imgA && imgB);
  const W = SCENE_WIDTH;

  return (
    <div className="matching-section">
      <div className="matching-topbar">
        <span className="matching-status" role="status">
          {error && `❌ ${error}`}
          {!error && !outcome && 'Click “Find Matches” to run Harris corners → rotation-normalised descriptors → RANSAC similarity estimation.'}
          {!error && outcome?.kind === 'ok' && `✅ ${outcome.pairs.length} matches, ${outcome.inliers} consistent with the estimated transform (${outcome.kpsA} / ${outcome.kpsB} corners).`}
          {!error && outcome?.kind === 'too-few' && `⚠ Only ${outcome.matches} match${outcome.matches === 1 ? '' : 'es'} found — not enough to estimate a transform. Try another transform.`}
        </span>
        <div className="matching-actions">
          <button type="button" className="btn btn-primary" onClick={findMatches} disabled={!ready}>
            {outcome ? '🔄 Re-run' : '🔍 Find Matches'}
          </button>
          <button type="button" className="btn btn-sm" onClick={newTransform}>🎲 New transform</button>
          <label className="matching-code-select">
            <span className="visually-hidden">Show code for</span>
            <select className="select" value="" onChange={(e) => setShowCode(e.target.value as keyof typeof CODE)}>
              <option value="" disabled>💻 Show code…</option>
              <option value="harris">harris.ts — corners</option>
              <option value="descriptors">descriptors.ts — matching</option>
              <option value="transform">transform.ts — RANSAC</option>
            </select>
          </label>
        </div>
      </div>

      <div className="matching-images">
        <div className="matching-panel">
          <h3>Image A (original)</h3>
          <PixelCanvas image={imgA?.imageData ?? null} size={W} pixelated={false} label="Original scene" />
        </div>
        <div className="matching-panel">
          <h3>Image B (transformed)</h3>
          <PixelCanvas image={imgB?.imageData ?? null} size={W} pixelated={false} label="Rotated, scaled and shifted scene" />
        </div>
        <div className="matching-panel matching-panel-wide">
          <h3>Matches <span className="legend-inlier">● inlier</span> <span className="legend-outlier">● outlier</span></h3>
          <PixelCanvas image={sideBySide} size={W * 2} pixelated={false} label="Both images side by side with match lines">
            {outcome?.kind === 'ok' && (
              <svg className="matching-overlay" viewBox={`0 0 ${W * 2} ${SCENE_HEIGHT}`} preserveAspectRatio="none" aria-hidden="true">
                {outcome.pairs.map((p, i) => (
                  <g key={i} className={p.inlier ? 'match-inlier' : 'match-outlier'}>
                    <line x1={p.a.x + 0.5} y1={p.a.y + 0.5} x2={p.b.x + W + 0.5} y2={p.b.y + 0.5} />
                    <circle cx={p.a.x + 0.5} cy={p.a.y + 0.5} r={2.2} />
                    <circle cx={p.b.x + W + 0.5} cy={p.b.y + 0.5} r={2.2} />
                  </g>
                ))}
              </svg>
            )}
          </PixelCanvas>
        </div>
      </div>

      {outcome?.kind === 'ok' && (
        <div className="matching-results">
          <div className="matching-result-grid">
            <div className="matching-stat">
              <span className="matching-stat-label">Matches</span>
              <span className="matching-stat-value">{outcome.pairs.length}</span>
            </div>
            <div className="matching-stat">
              <span className="matching-stat-label">RANSAC inliers</span>
              <span className="matching-stat-value">{outcome.inliers}</span>
            </div>
            <div className="matching-stat">
              <span className="matching-stat-label">Transform model</span>
              <span className="matching-stat-value">Similarity</span>
            </div>
          </div>
          <div className="matching-transform">
            <h4>Transform (A → B)</h4>
            <div className="matching-transform-row">
              <div className="matching-transform-col">
                <span className="matching-transform-label">Estimated</span>
                <pre className="matching-matrix">{formatSimilarity(outcome.transform)}</pre>
              </div>
              <div className="matching-transform-col">
                <span className="matching-transform-label">Ground truth</span>
                <pre className="matching-matrix matching-matrix-gt">{formatSimilarity(gt)}</pre>
              </div>
            </div>
            <p className="matching-transform-desc">
              RANSAC repeatedly fits the transform to 2 random matches and keeps the fit that most other matches agree
              with (within 3 px); the final estimate is a least-squares fit to those {outcome.inliers} inliers. Wrong matches
              (red) are ignored instead of pulling the estimate off.
            </p>
          </div>
        </div>
      )}

      {showCode && <CodeViewer code={CODE[showCode].code} title={CODE[showCode].title} onClose={() => setShowCode(null)} />}
    </div>
  );
}
