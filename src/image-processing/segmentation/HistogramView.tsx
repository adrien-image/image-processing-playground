import { useEffect, useRef } from 'react';
import { cssColor, useTheme } from '../../lib/theme';
import './HistogramView.css';

interface HistogramViewProps {
  /** Display histogram (any number of bins spanning gray levels 0–255). */
  histogram: Uint32Array;
  /** User threshold t (0–255): bins with values ≤ t are drawn as background. */
  threshold?: number;
  /** Otsu's threshold, drawn as a dashed marker. */
  otsuThreshold?: number;
  width?: number;
  height?: number;
  label: string;
}

const BACKGROUND = [108, 140, 255];
const FOREGROUND = [248, 113, 113];

export function HistogramView({ histogram, threshold, otsuThreshold, width = 360, height = 160, label }: HistogramViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const theme = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    // Render at device resolution so bars and labels stay sharp on HiDPI screens.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const bins = histogram.length;
    const pad = { top: 14, right: 8, bottom: 20, left: 8 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    /** x position of the boundary just after gray level v. */
    const xOfLevel = (v: number) => pad.left + ((v + 1) / 256) * plotW;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = cssColor('var(--chart-bg)');
    ctx.fillRect(0, 0, width, height);

    let maxFreq = 0;
    for (let i = 0; i < bins; i++) if (histogram[i] > maxFreq) maxFreq = histogram[i];
    if (maxFreq === 0) return;

    const barW = plotW / bins;
    for (let i = 0; i < bins; i++) {
      const h = (histogram[i] / maxFreq) * plotH;
      const x = pad.left + i * barW;
      const y = pad.top + plotH - h;
      // A bin covers gray levels [start, end]; colour it by which side its centre falls on.
      const centre = ((i + 0.5) * 256) / bins - 0.5;
      const [r, g, b] = threshold === undefined || centre <= threshold ? BACKGROUND : FOREGROUND;
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 + (h / plotH) * 0.5})`;
      ctx.fillRect(x, y, Math.max(1, barW - 1), h);
    }

    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';

    if (otsuThreshold !== undefined) {
      const x = xOfLevel(otsuThreshold);
      const otsuColor = cssColor('var(--success)');
      ctx.strokeStyle = otsuColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = otsuColor;
      ctx.fillText('Otsu', x, pad.top - 3);
    }

    if (threshold !== undefined && threshold !== otsuThreshold) {
      const x = xOfLevel(threshold);
      const userColor = cssColor('var(--warning)');
      ctx.strokeStyle = userColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
      ctx.fillStyle = userColor;
      ctx.fillText('You', x, pad.top + plotH + 12);
    }

    ctx.fillStyle = cssColor('var(--text-muted)');
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', pad.left, pad.top + plotH + 14);
    ctx.textAlign = 'right';
    ctx.fillText('255', pad.left + plotW, pad.top + plotH + 14);
    ctx.fillText(`${bins} bins`, width - pad.right, pad.top - 3);
  }, [histogram, threshold, otsuThreshold, width, height, theme]);

  return <canvas ref={canvasRef} className="histogram-canvas" style={{ maxWidth: width, aspectRatio: `${width} / ${height}` }} role="img" aria-label={label} />;
}
