import type { CSSProperties } from 'react';
interface KernelMatrixDisplayProps {
  kernel: number[][];
  label: string;
}

/** Displays a square kernel as a small grid — used in the sidebar. */
export function KernelMatrixDisplay({ kernel, label }: KernelMatrixDisplayProps) {
  return (
    <div className="kernel-matrix">
      <div className="kernel-matrix-header">{label}</div>
      <div className="kernel-matrix-grid" style={{ '--cols': kernel.length } as CSSProperties}>
        {kernel.flat().map((v, i) => (
          <span key={i} className="kernel-matrix-cell">
            {Number.isInteger(v) ? v : v.toFixed(3)}
          </span>
        ))}
      </div>
    </div>
  );
}
