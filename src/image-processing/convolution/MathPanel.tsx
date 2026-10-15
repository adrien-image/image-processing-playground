import { gridVars } from './gridSize';
import type { StepMath } from './playback';

/** Up to 2 decimals, without trailing zeros (51.000000001 → "51", 0.0625 → "0.06"). */
const fmt = (v: number) => String(Math.round(v * 100) / 100);

/** The element-wise multiply-and-sum behind the current output pixel. */
export function MathPanel({ data, outputMode }: { data: StepMath; outputMode: 'magnitude' | 'clamp' }) {
  const { neighbourhood, kernel, products, sum, output } = data;
  const size = kernel.length;
  const flatNb = neighbourhood.flat();
  const flatK = kernel.flat();
  const expression = flatNb.map((n, i) => `(${fmt(n)} × ${fmt(flatK[i])})`).join(' + ') + ` = ${fmt(sum)}`;
  const gridStyle = gridVars(size, 44);

  return (
    <div className="math-panel">
      <div className="math-panel-title">Convolution computation</div>
      <div className="math-panel-row">
        <div className="math-block">
          <div className="math-block-label">Neighbourhood</div>
          <div className="math-grid" style={gridStyle}>
            {flatNb.map((v, i) => <span key={i} className="math-cell math-cell-nb">{Math.round(v)}</span>)}
          </div>
        </div>
        <span className="math-op" aria-hidden="true">×</span>
        <div className="math-block">
          <div className="math-block-label">Kernel</div>
          <div className="math-grid" style={gridStyle}>
            {flatK.map((v, i) => <span key={i} className="math-cell math-cell-kernel">{fmt(v)}</span>)}
          </div>
        </div>
        <span className="math-op" aria-hidden="true">=</span>
        <div className="math-block">
          <div className="math-block-label">Products</div>
          <div className="math-grid" style={gridStyle}>
            {products.flat().map((v, i) => <span key={i} className="math-cell math-cell-prod">{fmt(v)}</span>)}
          </div>
        </div>
      </div>

      <div className="math-expression">{expression}</div>

      <div className="math-result">
        <div className="math-result-item">
          <span className="math-result-label">Σ (sum)</span>
          <span className="math-result-value">{fmt(sum)}</span>
        </div>
        <div className="math-result-item">
          <span className="math-result-label">
            Output {outputMode === 'magnitude' ? '(|Σ| scaled to the image max)' : '(Σ clamped to 0–255)'}
          </span>
          <span className="math-result-value math-result-output">{output}</span>
        </div>
      </div>
      <p className="math-note">
        Pixels outside the image repeat the nearest edge pixel (replicate padding). Strictly speaking this is
        cross-correlation — the kernel is not flipped — which is what image libraries and CNNs call “convolution”.
      </p>
    </div>
  );
}
