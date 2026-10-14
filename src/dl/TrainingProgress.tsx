import { useEffect, useMemo, useRef } from 'react';
import type { TrainingEpochData } from './lenet';

interface TrainingProgressProps {
  history: TrainingEpochData[];
  /** Epochs requested in the current run (for the "Epoch x / y" label) */
  runEpochs: number;
  /** Epochs completed in the current run */
  runDone: number;
  status: 'loading' | 'training' | 'done';
  batchProgress?: { current: number; total: number } | null;
}

const W = 440;
const H = 220;
const PAD = { top: 24, right: 44, bottom: 30, left: 44 };

const COLORS = {
  loss: '#f87171',
  valLoss: '#fb923c',
  acc: '#34d399',
  valAcc: '#6ee7b7',
};

/** "−0.1234" when the value went down, "+0.0123" when it went up. */
const signed = (v: number, digits: number, suffix = '') => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(v).toFixed(digits)}${suffix}`;

export function TrainingProgress({ history, runEpochs, runDone, status, batchProgress }: TrainingProgressProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stats = useMemo(() => {
    if (history.length === 0) return null;
    const last = history[history.length - 1];
    let best = history[0];
    for (const h of history) if (h.valAccuracy > best.valAccuracy) best = h;
    return {
      last,
      best,
      lossChange: last.loss - history[0].loss,
      accChange: last.accuracy - history[0].accuracy,
    };
  }, [history]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || history.length < 2) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.fillRect(0, 0, W, H);

    let maxLoss = 0.1;
    for (const d of history) maxLoss = Math.max(maxLoss, d.loss, d.valLoss);
    const toX = (i: number) => PAD.left + (i / (history.length - 1)) * plotW;
    const toYLoss = (v: number) => PAD.top + plotH - (v / maxLoss) * plotH;
    const toYAcc = (v: number) => PAD.top + plotH - v * plotH; // accuracy axis is fixed 0–100%

    // Grid + both y-axes (loss on the left, accuracy on the right)
    ctx.font = '10px monospace';
    for (let i = 0; i <= 4; i++) {
      const y = PAD.top + (i / 4) * plotH;
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
      ctx.fillStyle = COLORS.loss;
      ctx.textAlign = 'right';
      ctx.fillText((maxLoss * (1 - i / 4)).toFixed(2), PAD.left - 5, y + 3);
      ctx.fillStyle = COLORS.acc;
      ctx.textAlign = 'left';
      ctx.fillText(`${100 - i * 25}%`, W - PAD.right + 5, y + 3);
    }

    // x-axis: epoch numbers
    ctx.fillStyle = '#6b728e';
    ctx.textAlign = 'center';
    const every = Math.ceil(history.length / 10);
    history.forEach((d, i) => {
      if (i % every === 0 || i === history.length - 1) ctx.fillText(String(d.epoch), toX(i), H - PAD.bottom + 14);
    });
    ctx.fillText('epoch', PAD.left + plotW / 2, H - 4);

    const curve = (values: number[], toY: (v: number) => number, color: string, dashed: boolean) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dashed ? [4, 4] : []);
      ctx.beginPath();
      values.forEach((v, i) => (i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v))));
      ctx.stroke();
      ctx.setLineDash([]);
    };
    curve(history.map(h => h.loss), toYLoss, COLORS.loss, false);
    curve(history.map(h => h.valLoss), toYLoss, COLORS.valLoss, true);
    curve(history.map(h => h.accuracy), toYAcc, COLORS.acc, false);
    curve(history.map(h => h.valAccuracy), toYAcc, COLORS.valAcc, true);

    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    [
      { label: 'Loss (left axis)', color: COLORS.loss, dashed: false },
      { label: 'Val loss', color: COLORS.valLoss, dashed: true },
      { label: 'Accuracy (right axis)', color: COLORS.acc, dashed: false },
      { label: 'Val acc', color: COLORS.valAcc, dashed: true },
    ].reduce((x, l) => {
      ctx.strokeStyle = l.color;
      ctx.lineWidth = 2;
      ctx.setLineDash(l.dashed ? [3, 2] : []);
      ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x + 14, 10); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#949bb5';
      ctx.fillText(l.label, x + 18, 13);
      return x + 26 + ctx.measureText(l.label).width;
    }, PAD.left);
  }, [history]);

  return (
    <div className="training-progress">
      <div className="training-progress-header">
        <h3>Training Progress</h3>
        <span className="training-status" role="status">
          {status === 'loading' && '⏳ Loading MNIST…'}
          {status === 'training' && `⏳ Epoch ${Math.min(runDone + 1, runEpochs)} / ${runEpochs}`}
          {status === 'done' && `✅ ${history.length} epoch${history.length === 1 ? '' : 's'} trained`}
        </span>
      </div>

      {status === 'training' && batchProgress && (
        <div className="training-batch-bar">
          <div className="training-batch-label">
            Batch {batchProgress.current} / {batchProgress.total}
            <span className="training-batch-pct">({Math.round((batchProgress.current / batchProgress.total) * 100)}%)</span>
          </div>
          <div className="training-batch-track">
            <div className="training-batch-fill" style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }} />
          </div>
        </div>
      )}

      {status === 'loading' && (
        <div className="training-loading">
          <div className="training-loading-bar" />
          <p>Loading 8,000 MNIST digits (≈1.2 MB): 6,000 for training, 1,000 for validation, 1,000 for testing…</p>
        </div>
      )}

      {history.length >= 2 && (
        <canvas ref={canvasRef} className="training-canvas" role="img"
          aria-label={`Loss and accuracy over ${history.length} epochs`} />
      )}

      {stats && (
        <div className="training-metrics">
          <div className="training-metrics-grid">
            <div className="training-metric">
              <span className="training-metric-label">Training Loss</span>
              <span className="training-metric-value">{stats.last.loss.toFixed(4)}</span>
              {history.length > 1 && <span className="training-metric-delta">{signed(stats.lossChange, 4)} since epoch 1</span>}
            </div>
            <div className="training-metric">
              <span className="training-metric-label">Training Acc</span>
              <span className="training-metric-value">{(stats.last.accuracy * 100).toFixed(1)}%</span>
              {history.length > 1 && <span className="training-metric-delta">{signed(stats.accChange * 100, 1, ' pts')}</span>}
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
              <span className="training-metric-value">{(stats.best.valAccuracy * 100).toFixed(1)}%</span>
              <span className="training-metric-delta">epoch {stats.best.epoch}</span>
            </div>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div className="training-epoch-table-wrap">
          <table className="training-epoch-table">
            <thead>
              <tr>
                <th scope="col">Epoch</th>
                <th scope="col">Loss</th>
                <th scope="col">Accuracy</th>
                <th scope="col">Val Loss</th>
                <th scope="col">Val Acc</th>
              </tr>
            </thead>
            <tbody>
              {history.map(h => {
                const isBest = stats?.best === h;
                return (
                  <tr key={h.epoch} className={isBest ? 'tr-best' : ''}>
                    <td>{h.epoch}{isBest ? ' ⭐' : ''}</td>
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
