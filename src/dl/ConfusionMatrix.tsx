interface ConfusionMatrixProps {
  matrix: number[][];
}

import { DIGITS } from './constants';

export function ConfusionMatrix({ matrix }: ConfusionMatrixProps) {
  // Find the maximum value for colour scaling
  let maxVal = 0;
  for (const row of matrix) {
    for (const v of row) {
      if (v > maxVal) maxVal = v;
    }
  }

  // Compute per-digit stats
  const totalPerDigit = matrix.map(row => row.reduce((a, b) => a + b, 0));
  const accuracyPerDigit = matrix.map((row, i) => totalPerDigit[i] > 0 ? row[i] / totalPerDigit[i] : 0);
  const mostConfused: { digit: number; confusedWith: number; count: number }[] = [];
  for (let i = 0; i < 10; i++) {
    let maxOff = 0, maxOffDigit = -1;
    for (let j = 0; j < 10; j++) {
      if (j !== i && matrix[i][j] > maxOff) { maxOff = matrix[i][j]; maxOffDigit = j; }
    }
    if (maxOffDigit >= 0) mostConfused.push({ digit: i, confusedWith: maxOffDigit, count: maxOff });
  }

  return (
    <div className="cm-container">
      <h3>Confusion Matrix</h3>

      <div className="cm-grid-wrap">
        <table className="cm-table">
          <thead>
            <tr>
              <th></th>
              {DIGITS.map(d => <th key={d} className="cm-col-label">{d}</th>)}
              <th className="cm-total-label">Total</th>
              <th className="cm-acc-label">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="cm-row-label">{DIGITS[i]}</td>
                {row.map((v, j) => {
                  const intensity = maxVal > 0 ? v / maxVal : 0;
                  const isDiagonal = i === j;
                  return (
                    <td
                      key={j}
                      className={`cm-cell${isDiagonal ? ' cm-cell-diag' : ''}${v > 0 && !isDiagonal ? ' cm-cell-off' : ''}`}
                      style={{
                        backgroundColor: isDiagonal
                          ? `rgba(52, 211, 153, ${0.15 + intensity * 0.6})`
                          : v > 0
                            ? `rgba(248, 113, 113, ${0.1 + intensity * 0.5})`
                            : 'transparent',
                      }}
                    >
                      {v > 0 ? v : ''}
                    </td>
                  );
                })}
                <td className="cm-total">{totalPerDigit[i]}</td>
                <td className="cm-acc">{(accuracyPerDigit[i] * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="cm-legend">
        <span><span className="cm-legend-swatch cm-legend-correct" /> Correct prediction</span>
        <span><span className="cm-legend-swatch cm-legend-wrong" /> Misclassification</span>
        <span className="cm-legend-text">Rows = true label · Columns = predicted label</span>
      </div>

      <div className="cm-interpretation">
        <h4>What is a Confusion Matrix?</h4>
        <p>
          A confusion matrix is a 10×10 table that shows how a classifier's predictions match the actual labels.
          Each row represents the <strong>true</strong> digit, and each column represents the digit the model <strong>predicted</strong>.
        </p>
        <p>
          <strong>Diagonal cells</strong> (green) show correct predictions — the higher the number, the better the model performs on that digit.
          <strong>Off-diagonal cells</strong> (red) show mistakes. For example, if cell (4, 9) has a high value, it means the model often confuses a "4" for a "9".
        </p>
        <p>
          The confusion matrix is computed by running all 1,000 test images through the trained model,
          comparing each predicted digit against the true label, and counting every (true, predicted) pair.
        </p>

        <h4>How it's computed</h4>
        <div className="cm-formula">
          <code>CM[trueLabel][predictedLabel] += 1</code>
          <span>for each test sample</span>
        </div>
        <p>
          Start with a 10×10 matrix of zeros. For each test image, find the true digit and the model's predicted digit
          (the class with the highest softmax probability), then increment the cell at (true, predicted).
          After all 1,000 images are processed, the matrix shows the distribution of predictions per digit.
        </p>

        {mostConfused.length > 0 && (
          <>
            <h4>Most common confusions</h4>
            <ul className="cm-confusions">
              {mostConfused
                .sort((a, b) => b.count - a.count)
                .slice(0, 5)
                .map((c, i) => (
                  <li key={i}>
                    Digit <strong>{DIGITS[c.digit]}</strong> was predicted as <strong>{DIGITS[c.confusedWith]}</strong> {c.count} time{c.count > 1 ? 's' : ''}
                  </li>
                ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
