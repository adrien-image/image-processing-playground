import { useId } from 'react';
import './LineChart.css';

export interface ChartSeries {
  label: string;
  color: string;
  values: readonly number[];
  dashed?: boolean;
}

interface LineChartProps {
  series: ChartSeries[];
  /** Accessible summary of what the chart shows. */
  title: string;
  xLabel?: string;
  /** Logarithmic y axis (values ≤ 0 are skipped). */
  logY?: boolean;
  /** Fixed y range; by default it fits the data. */
  yMin?: number;
  yMax?: number;
  /** x value of index 0 (e.g. 1 for epochs). */
  xStart?: number;
  formatY?: (v: number) => string;
  height?: number;
  /** A vertical marker at this x index. */
  markerX?: number;
  /** Label for the tick at index i (default: i + xStart). */
  xTickLabel?: (i: number) => string;
}

const W = 480;
const PAD = { top: 10, right: 12, bottom: 26, left: 48 };

const defaultFormat = (v: number) => {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000 || a < 0.01) return v.toExponential(0);
  return String(Number(v.toPrecision(2)));
};

/**
 * A small responsive line chart (SVG, scales to its container's width) for training
 * curves and other per-step values. Series may have different lengths.
 */
export function LineChart({
  series, title, xLabel, logY = false, yMin, yMax, xStart = 0, formatY = defaultFormat, height = 200, markerX, xTickLabel,
}: LineChartProps) {
  const titleId = useId();
  const H = height;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const n = Math.max(2, ...series.map(s => s.values.length));

  const t = (v: number) => (logY ? Math.log10(v) : v);
  const usable = series.flatMap(s => s.values.filter(v => Number.isFinite(v) && (!logY || v > 0)));
  let lo = yMin !== undefined ? t(yMin) : usable.length ? Math.min(...usable.map(t)) : 0;
  let hi = yMax !== undefined ? t(yMax) : usable.length ? Math.max(...usable.map(t)) : 1;
  if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
  if (logY) { lo = Math.floor(lo); hi = Math.ceil(hi); }

  const x = (i: number) => PAD.left + (i / (n - 1)) * plotW;
  const y = (v: number) => PAD.top + (1 - (t(v) - lo) / (hi - lo)) * plotH;

  const ticks = logY
    ? Array.from({ length: hi - lo + 1 }, (_, k) => 10 ** (lo + k))
    : Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);
  const xTicks = [...new Set([0, Math.round((n - 1) / 2), n - 1])];

  const path = (values: readonly number[]) => {
    let d = '';
    let pen = false;
    values.forEach((v, i) => {
      if (!Number.isFinite(v) || (logY && v <= 0)) { pen = false; return; }
      const yy = Math.min(H - PAD.bottom, Math.max(PAD.top, y(v)));
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${yy.toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  return (
    <figure className="line-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={titleId} preserveAspectRatio="xMidYMid meet">
        <title id={titleId}>{title}</title>
        {ticks.map(v => (
          <g key={v}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} className="line-chart-grid" />
            <text x={PAD.left - 6} y={y(v)} className="line-chart-tick" textAnchor="end" dominantBaseline="middle">{formatY(v)}</text>
          </g>
        ))}
        {xTicks.map(i => (
          <text key={i} x={x(i)} y={H - PAD.bottom + 12} className="line-chart-tick" textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}>{xTickLabel ? xTickLabel(i) : i + xStart}</text>
        ))}
        {xLabel && <text x={PAD.left + plotW / 4} y={H - 2} className="line-chart-tick line-chart-xlabel" textAnchor="middle">{xLabel} →</text>}
        {markerX !== undefined && (
          <line x1={x(markerX)} x2={x(markerX)} y1={PAD.top} y2={H - PAD.bottom} className="line-chart-marker" />
        )}
        {series.map(s => (
          <path key={s.label} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2}
            strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
        ))}
      </svg>
      <figcaption className="line-chart-legend">
        {series.map(s => (
          <span key={s.label} className="line-chart-key">
            <span className={`line-chart-swatch${s.dashed ? ' line-chart-swatch-dashed' : ''}`} style={{ borderColor: s.color }} />
            {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
