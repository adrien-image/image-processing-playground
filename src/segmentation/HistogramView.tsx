import { useRef, useEffect } from 'react';

interface HistogramViewProps {
  histogram: Uint32Array;
  threshold: number;
  otsuThreshold: number;
  width?: number;
  height?: number;
}

export function HistogramView({ histogram, threshold, otsuThreshold, width = 360, height = 160 }: HistogramViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = width;
    canvas.height = height;

    const bins = histogram.length;
    const pad = { top: 8, right: 8, bottom: 20, left: 8 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Clear
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(0, 0, width, height);

    // Find max frequency
    let maxFreq = 0;
    for (let i = 0; i < bins; i++) {
      if (histogram[i] > maxFreq) maxFreq = histogram[i];
    }
    if (maxFreq === 0) return;

    const barW = plotW / bins;

    // Draw bars
    for (let i = 0; i < bins; i++) {
      const h = (histogram[i] / maxFreq) * plotH;
      const x = pad.left + i * barW;
      const y = pad.top + plotH - h;

      // Colour bins — single neutral colour when threshold < 0, else split at threshold
      const binVal = i / bins;
      if (threshold < 0) {
        ctx.fillStyle = `rgba(108, 140, 255, ${0.25 + (h / plotH) * 0.55})`;
      } else if (binVal <= threshold) {
        ctx.fillStyle = `rgba(108, 140, 255, ${0.3 + (h / plotH) * 0.5})`;
      } else {
        ctx.fillStyle = `rgba(248, 113, 113, ${0.3 + (h / plotH) * 0.5})`;
      }
      ctx.fillRect(x, y, Math.max(1, barW - 1), h);
    }

    // Otsu threshold line (optional)
    // Otsu threshold line
    if (otsuThreshold >= 0) {
      const otsuX = pad.left + otsuThreshold * plotW;
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(otsuX, pad.top);
      ctx.lineTo(otsuX, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#34d399';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Otsu', otsuX, pad.top - 2);
    }

    // User threshold line (if different from Otsu)
    const userX = pad.left + threshold * plotW;
    if (threshold >= 0 && Math.abs(threshold - otsuThreshold) > 0.01) {
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(userX, pad.top);
      ctx.lineTo(userX, pad.top + plotH);
      ctx.stroke();

      ctx.fillStyle = '#fbbf24';
      ctx.fillText('User', userX, pad.top + plotH + 12);
    }

    // X-axis labels
    ctx.fillStyle = '#6b728e';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('0', pad.left, pad.top + plotH + 14);
    ctx.fillText('255', pad.left + plotW, pad.top + plotH + 14);
    
    // Show bin count badge
    ctx.textAlign = 'right';
    ctx.fillStyle = '#6b728e';
    ctx.font = '8px sans-serif';
    ctx.fillText(`${bins} bins`, width - pad.right, pad.top + plotH + 14);
  }, [histogram, threshold, otsuThreshold, width, height]);

  return <canvas ref={canvasRef} className="histogram-canvas" />;
}
