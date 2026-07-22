interface KernelMatrixDisplayProps {
  kernel: number[][];
  label: string;
}

/** Displays any 3×3 kernel as a small grid — used in the sidebar. */
export function KernelMatrixDisplay({ kernel, label }: KernelMatrixDisplayProps) {
  return (
    <div className="sobel-matrix">
      <div className="sobel-matrix-header">{label}</div>
      <div className="sobel-matrix-grid">
        {kernel.flat().map((v, i) => (
          <span key={i} className="sobel-matrix-cell">
            {Number.isInteger(v) ? v : v.toFixed(3)}
          </span>
        ))}
      </div>
    </div>
  );
}
