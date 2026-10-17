import { useEffect, useRef, type PointerEvent } from 'react';
import { cssColor, useTheme } from '../../lib/theme';
import { CLASS_COLORS, UNIT_EXTENT, type ClassMap, type Extent } from './plot';

export interface PointLayer {
  xy: Float32Array;
  /** Colour each point by its class; without labels every point gets `color`. */
  labels?: ArrayLike<number>;
  color?: string;
  size?: number;
  shape?: 'dot' | 'ring' | 'cross';
  /** Only draw the points where mask[i] is truthy. */
  mask?: ArrayLike<number | boolean>;
}

/** What an overlay receives to draw lines, ellipses or labels in data coordinates. */
export interface Draw {
  ctx: CanvasRenderingContext2D;
  toPx: (x: number, y: number) => [number, number];
  /** Length in pixels of `d` data units along x. */
  scale: number;
  color: (css: string) => string;
  size: number;
}

interface Props {
  layers: PointLayer[];
  map?: ClassMap | null;
  extent?: Extent;
  size?: number;
  label: string;
  overlay?: (d: Draw) => void;
  /** Called with the data coordinates of a click or tap. */
  onPick?: (x: number, y: number, e: PointerEvent<HTMLCanvasElement>) => void;
}

/** A square scatter plot over an optional decision map, redrawn when the theme changes. */
export function ScatterPlot({ layers, map, extent = UNIT_EXTENT, size = 360, label, overlay, onPick }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [x0, x1] = extent.x, [y0, y1] = extent.y;
    const toPx = (x: number, y: number): [number, number] => [((x - x0) / (x1 - x0)) * size, ((y1 - y) / (y1 - y0)) * size];
    const colors = CLASS_COLORS.map(c => rgb(cssColor(c)));

    ctx.fillStyle = cssColor('var(--chart-bg)');
    ctx.clearRect(0, 0, size, size);
    ctx.fillRect(0, 0, size, size);
    if (map) {
      // Each cell mixes the class colours by probability; confident regions are more opaque.
      const img = new ImageData(map.res, map.res);
      const k = map.classes;
      for (let i = 0; i < map.res * map.res; i++) {
        let r = 0, g = 0, b = 0, top = 0;
        for (let c = 0; c < k; c++) {
          const p = map.probs[i * k + c], col = colors[c % colors.length];
          r += p * col[0]; g += p * col[1]; b += p * col[2]; top = Math.max(top, p);
        }
        const confidence = k > 1 ? (top - 1 / k) / (1 - 1 / k) : top;
        img.data.set([r, g, b, 255 * (0.12 + 0.3 * confidence)], i * 4);
      }
      const off = document.createElement('canvas');
      off.width = off.height = map.res;
      off.getContext('2d')?.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, size, size);
    }
    // Axes through the origin, when it is in view.
    ctx.strokeStyle = cssColor('color-mix(in srgb, var(--fg) 12%, transparent)');
    ctx.lineWidth = 1;
    const [ox, oy] = toPx(0, 0);
    ctx.beginPath();
    if (x0 < 0 && x1 > 0) { ctx.moveTo(ox, 0); ctx.lineTo(ox, size); }
    if (y0 < 0 && y1 > 0) { ctx.moveTo(0, oy); ctx.lineTo(size, oy); }
    ctx.stroke();

    const outline = theme === 'light' ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.7)';
    for (const layer of layers) {
      const s = layer.size ?? 5;
      const fixed = layer.color ? cssColor(layer.color) : null;
      for (let i = 0; i < layer.xy.length / 2; i++) {
        if (layer.mask && !layer.mask[i]) continue;
        const [px, py] = toPx(layer.xy[2 * i], layer.xy[2 * i + 1]);
        const c = fixed ?? `rgb(${colors[(layer.labels?.[i] ?? 0) % colors.length].join(',')})`;
        ctx.beginPath();
        if (layer.shape === 'cross') {
          ctx.strokeStyle = c; ctx.lineWidth = 2;
          ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s); ctx.moveTo(px - s, py + s); ctx.lineTo(px + s, py - s);
          ctx.stroke();
        } else if (layer.shape === 'ring') {
          ctx.strokeStyle = c; ctx.lineWidth = 2;
          ctx.arc(px, py, s, 0, 2 * Math.PI);
          ctx.stroke();
        } else {
          ctx.fillStyle = c; ctx.strokeStyle = outline; ctx.lineWidth = 1;
          ctx.arc(px, py, s / 2, 0, 2 * Math.PI);
          ctx.fill(); ctx.stroke();
        }
      }
    }
    overlay?.({ ctx, toPx, scale: size / (x1 - x0), color: cssColor, size });
  }, [layers, map, extent, size, overlay, theme]);

  const pick = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!onPick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width, fy = (e.clientY - rect.top) / rect.height;
    onPick(extent.x[0] + fx * (extent.x[1] - extent.x[0]), extent.y[1] - fy * (extent.y[1] - extent.y[0]), e);
  };

  return (
    <canvas ref={ref} className={`ml-plot${onPick ? ' ml-plot-pick' : ''}`} style={{ width: size, maxWidth: '100%', aspectRatio: '1' }}
      role="img" aria-label={label} onPointerDown={onPick ? pick : undefined} />
  );
}

function rgb(color: string): [number, number, number] {
  const m = color.match(/[\d.]+/g);
  return m ? [Number(m[0]), Number(m[1]), Number(m[2])] : [128, 128, 128];
}
