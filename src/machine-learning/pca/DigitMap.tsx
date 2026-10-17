import { useCallback, useMemo } from 'react';
import { ScatterPlot, type Draw, type PointLayer } from '../shared/ScatterPlot';
import type { Extent } from '../shared/plot';
import { DIGIT_COLORS, fitExtent } from './digits';

const NO_LAYERS: PointLayer[] = [];

interface Props {
  /** n (x, y) pairs. */
  xy: Float32Array | Float64Array;
  labels: ArrayLike<number>;
  label: string;
  size?: number;
  /** Draw each point as its digit (default) or as a small dot. */
  glyphs?: boolean;
  extent?: Extent;
}

/**
 * A map of digits: every point drawn as its own digit, in that digit's colour, so the
 * map stays readable without relying on colour alone. The extent fits the points.
 */
export function DigitMap({ xy, labels, label, size = 360, glyphs = true, extent }: Props) {
  const fitted = useMemo(() => extent ?? fitExtent(xy), [extent, xy]);
  const overlay = useCallback(({ ctx, toPx, color }: Draw) => {
    const colors = DIGIT_COLORS.map(c => color(c));
    const font = getComputedStyle(ctx.canvas).fontFamily || 'sans-serif';
    ctx.font = `600 ${glyphs ? 10 : 9}px ${font}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha = 0.85;
    for (let i = 0; i < labels.length; i++) {
      const [px, py] = toPx(xy[2 * i], xy[2 * i + 1]);
      ctx.fillStyle = colors[labels[i]];
      if (glyphs) ctx.fillText(String(labels[i]), px, py);
      else {
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }, [xy, labels, glyphs]);
  return <ScatterPlot layers={NO_LAYERS} extent={fitted} size={size} label={label} overlay={overlay} />;
}

/** The colour key for DigitMap. */
export function DigitLegend() {
  return (
    <div className="ml-legend" aria-label="Colour of each digit">
      {DIGIT_COLORS.map((c, d) => <span key={d}><i className="ml-swatch" style={{ background: c }} />{d}</span>)}
    </div>
  );
}
