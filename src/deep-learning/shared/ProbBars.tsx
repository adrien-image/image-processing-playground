import './dl-shared.css';

interface ProbBarsProps {
  probs: ArrayLike<number>;
  /** The true class, outlined. */
  trueLabel?: number;
  title: string;
}

/** Class probabilities for digits 0–9, in digit order so two charts line up. */
export function ProbBars({ probs, trueLabel, title }: ProbBarsProps) {
  let top = 0;
  for (let d = 1; d < 10; d++) if (probs[d] > probs[top]) top = d;
  return (
    <div className="prob-bars">
      <div className="prob-bars-title">{title}: <strong>{top}</strong> ({(probs[top] * 100).toFixed(1)}%)</div>
      <div className="prob-bars-grid" role="list">
        {Array.from({ length: 10 }, (_, d) => (
          <div key={d} role="listitem"
            className={`prob-bar${d === top ? ' prob-bar-top' : ''}${d === trueLabel ? ' prob-bar-true' : ''}`}
            aria-label={`${d}: ${(probs[d] * 100).toFixed(1)}%`}>
            <div className="prob-bar-track"><div className="prob-bar-fill" style={{ height: `${probs[d] * 100}%` }} /></div>
            <span className="prob-bar-label">{d}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
