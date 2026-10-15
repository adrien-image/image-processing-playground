import { gridVars } from './gridSize';

interface KernelMatrixDisplayProps {
  kernel: number[][];
  label: string;
}

/** Kernels larger than this are drawn as a heat map: their numbers would not fit the sidebar. */
const MAX_NUMBERED = 5;

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : (Math.round(v * 100) / 100).toFixed(2));

/** Displays a square kernel as a small grid — used in the sidebar. */
export function KernelMatrixDisplay({ kernel, label }: KernelMatrixDisplayProps) {
  const size = kernel.length;
  const heatmap = size > MAX_NUMBERED;
  const max = Math.max(...kernel.flat().map(Math.abs)) || 1;

  return (
    <div className="kernel-matrix">
      <div className="kernel-matrix-header">{label} · {size}×{size}</div>
      <div className="kernel-matrix-grid" style={gridVars(size, 40)}>
        {kernel.flat().map((v, i) => (
          <span
            key={i}
            className={`kernel-matrix-cell${heatmap ? ' kernel-matrix-cell-heat' : ''}`}
            style={heatmap ? { opacity: 0.15 + 0.85 * (Math.abs(v) / max) } : undefined}
            title={heatmap ? fmt(v) : undefined}
          >
            {heatmap ? null : fmt(v)}
          </span>
        ))}
      </div>
      {heatmap && (
        <p className="kernel-matrix-note">Brighter cells weigh more. Hover a cell for its value; every weight is in the computation below.</p>
      )}
    </div>
  );
}
