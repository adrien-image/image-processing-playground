import { useRef, useEffect, useMemo } from 'react';
import type { TrainingEpochData } from './lenet';

interface TrainingProgressProps {
  history: TrainingEpochData[];
  currentEpoch: number;
  totalEpochs: number;
  status: 'idle' | 'loading' | 'training' | 'done' | 'error';
  batchProgress?: { current: number; total: number } | null;
}

const W = 420;
const H = 200;
const PAD = { top: 20, right: 20, bottom: 30, left: 45 };

export function TrainingProgress({ history, currentEpoch, totalEpochs, status, batchProgress }: TrainingProgressProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Compute derived stats
  const stats = useMemo(() => {
    if (history.length === 0) return null;
    const last = history[history.length - 1];
    const bestAcc = Math.max(...history.map(h => h.valAccuracy * 100));
    const bestEpoch = history.findIndex(h => h.valAccuracy * 100 === bestAcc) + 1;
    const lossImprovement = history.length > 1 ? history[0].loss - last.loss : 0;
    const accImprovement = history.length > 1 ? last.accuracy - history[0].accuracy : 0;
    return { last, bestAcc, bestEpoch, lossImprovement, accImprovement };
  }, [history]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = W;
    canvas.height = H;

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(0, 0, W, H);

    let maxLoss = 0, maxAcc = 0;
    for (const d of history) {
      if (d.loss > maxLoss) maxLoss = d.loss;
      if (d.valLoss > maxLoss) maxLoss = d.valLoss;
      if (d.accuracy > maxAcc) maxAcc = d.accuracy;
      if (d.valAccuracy > maxAcc) maxAcc = d.valAccuracy;
    }
    maxLoss = Math.max(maxLoss, 0.1);
    maxAcc = Math.max(maxAcc, 0.1);

    const toX = (i: number) => PAD.left + (i / Math.max(history.length - 1, 1)) * plotW;
    const toYLoss = (v: number) => PAD.top + plotH - (v / maxLoss) * plotH * 0.9;
    const toYAcc = (v: number) => PAD.top + plotH - (v / maxAcc) * plotH * 0.9;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (i / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = '#6b728e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (i / 4) * plotH;
      ctx.fillText((maxLoss * (1 - i / 4)).toFixed(2), PAD.left - 5, y + 3);
    }

    // X-axis labels (epoch numbers)
    ctx.textAlign = 'center';
    for (let i = 0; i < history.length; i++) {
      if (history.length <= 10 || i === 0 || i === history.length - 1 || i % Math.ceil(history.length / 5) === 0) {
        ctx.fillText(`${i + 1}`, toX(i), H - 8);
      }
    }

    // Draw curves
    const drawCurve = (data: number[], color: string, dashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([4, 4]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = toX(i); const y = toYLoss(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };
    const drawAccCurve = (data: number[], color: string, dashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([4, 4]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = toX(i); const y = toYAcc(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    drawCurve(history.map(h => h.loss), '#f87171');
    drawCurve(history.map(h => h.valLoss), '#fb923c', true);
    drawAccCurve(history.map(h => h.accuracy), '#34d399');
    drawAccCurve(history.map(h => h.valAccuracy), '#6ee7b7', true);

    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    [
      { label: 'Loss', color: '#f87171' },
      { label: 'Val Loss', color: '#fb923c' },
      { label: 'Accuracy', color: '#34d399' },
      { label: 'Val Acc', color: '#6ee7b7' },
    ].forEach((l, i) => {
      const x = PAD.left + i * 90;
      ctx.fillStyle = l.color;
      ctx.globalAlpha = i % 2 === 1 ? 0.7 : 1;
      ctx.fillRect(x, 8, 14, 2);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#949bb5';
      ctx.fillText(l.label, x + 18, 12);
    });

    // Latest values
    const last = history[history.length - 1];
    ctx.textAlign = 'right';
    ctx.fillStyle = '#e8ecf4';
    ctx.font = '11px monospace';
    ctx.fillText(`loss: ${last.loss.toFixed(4)}  acc: ${(last.accuracy * 100).toFixed(1)}%`, W - PAD.right, H - 6);
    ctx.fillText(`val:   ${last.valLoss.toFixed(4)}  val: ${(last.valAccuracy * 100).toFixed(1)}%`, W - PAD.right, H - 6 + 14);
  }, [history]);

  return (
    <div className="training-progress">
      <div className="training-progress-header">
        <h3>Training Progress</h3>
        <span className="training-status">
          {status === 'loading' && '⏳ Downloading MNIST (60k images)…'}
          {status === 'training' && `⏳ Epoch ${currentEpoch} / ${totalEpochs}`}
          {status === 'done' && `✅ Complete — ${history.length} epoch${history.length > 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Batch progress bar (during training) */}
      {status === 'training' && batchProgress && (
        <div className="training-batch-bar">
          <div className="training-batch-label">
            Batch {batchProgress.current} / {batchProgress.total}
            <span className="training-batch-pct">
              ({Math.round((batchProgress.current / batchProgress.total) * 100)}%)
            </span>
          </div>
          <div className="training-batch-track">
            <div
              className="training-batch-fill"
              style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {status === 'loading' && (
        <div className="training-loading">
          <div className="training-loading-bar" />
          <p>Loading 60,000 training images + 10,000 test images from Google Cloud…</p>
        </div>
      )}

      {history.length >= 2 && <canvas ref={canvasRef} className="training-canvas" />}

      {/* Stats cards */}
      {stats && (
        <div className="training-metrics">
          <div className="training-metrics-grid">
            <div className="training-metric">
              <span className="training-metric-label">Final Training Loss</span>
              <span className="training-metric-value">{stats.last.loss.toFixed(4)}</span>
              <span className="training-metric-delta">{(stats.lossImprovement > 0 ? '-': '') + stats.lossImprovement.toFixed(4)}</span>
            </div>
            <div className="training-metric">
              <span className="training-metric-label">Final Training Acc</span>
              <span className="training-metric-value">{(stats.last.accuracy * 100).toFixed(1)}%</span>
              <span className="training-metric-delta">+{(stats.accImprovement * 100).toFixed(1)}%</span>
            </div>
            <div className="training-metric">
              <span className="training-metric-label">Val Loss</span>
              <span className="training-metric-value">{stats.last.valLoss.toFixed(4)}</span>
            </div>
            <div className="training-metric">
              <span className="training-metric-label">Val Accuracy</span>
              <span className="training-metric-value">{(stats.last.valAccuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="training-metric training-metric-best">
              <span className="training-metric-label">⭐ Best Val Accuracy</span>
              <span className="training-metric-value">{stats.bestAcc.toFixed(1)}%</span>
              <span className="training-metric-delta">epoch {stats.bestEpoch}</span>
            </div>
          </div>
        </div>
      )}

      {/* Epoch table */}
      {history.length > 0 && (
        <div className="training-epoch-table-wrap">
          <table className="training-epoch-table">
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Loss</th>
                <th>Accuracy</th>
                <th>Val Loss</th>
                <th>Val Acc</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h, i) => {
                const isBest = stats && h.valAccuracy * 100 === stats.bestAcc;
                return (
                  <tr key={i} className={isBest ? 'tr-best' : ''}>
                    <td>{i + 1}{isBest ? ' ⭐' : ''}</td>
                    <td>{h.loss.toFixed(4)}</td>
                    <td>{(h.accuracy * 100).toFixed(1)}%</td>
                    <td>{h.valLoss.toFixed(4)}</td>
                    <td>{(h.valAccuracy * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
