import { DIGITS } from './constants';

interface PredictionDisplayProps {
  predictions: Float32Array | null;
}

export function PredictionDisplay({ predictions }: PredictionDisplayProps) {
  if (!predictions) {
    return (
      <div className="prediction-display">
        <h3>Prediction</h3>
        <p className="prediction-placeholder">Draw a digit and click Classify to see the prediction.</p>
      </div>
    );
  }

  const sorted = Array.from(predictions)
    .map((prob, digit) => ({ digit, prob }))
    .sort((a, b) => b.prob - a.prob);
  const top = sorted[0];

  return (
    <div className="prediction-display">
      <h3>Prediction</h3>
      <div className="prediction-top" aria-live="polite">
        <span className="prediction-digit">{DIGITS[top.digit]}</span>
        <span className="prediction-confidence">{(top.prob * 100).toFixed(1)}%</span>
      </div>

      <div className="prediction-bars">
        {sorted.map(({ digit, prob }) => (
          <div key={digit} className={`prediction-bar-row${digit === top.digit ? ' prediction-bar-top' : ''}`}>
            <span className="prediction-bar-label">{DIGITS[digit]}</span>
            <div className="prediction-bar-track">
              <div className="prediction-bar-fill" style={{ width: `${top.prob > 0 ? (prob / top.prob) * 100 : 0}%` }} />
            </div>
            <span className="prediction-bar-prob">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
