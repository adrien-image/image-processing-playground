import type { CSSProperties } from 'react';

/**
 * CSS variables for a kernel-sized grid of number cells: cells (and their text)
 * shrink as the kernel grows, so a 9×9 kernel still fits where a 3×3 one does.
 */
export function gridVars(size: number, cell3: number): CSSProperties {
  const scale = size <= 3 ? 1 : size <= 5 ? 0.8 : size <= 7 ? 0.68 : 0.58;
  return {
    '--cols': size,
    '--cell': `${Math.round(cell3 * scale)}px`,
    '--cell-font': `${(0.78 * Math.max(scale, 0.75)).toFixed(3)}rem`,
  } as CSSProperties;
}
