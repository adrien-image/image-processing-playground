import { useRef, useEffect } from 'react';
import type { TestExample } from './lenet';

interface TestResultsProps {
  examples: TestExample[];
  accuracy: number;
}

import { DIGITS } from './constants';

function ExampleCard({ example }: { example: TestExample }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(example.pixels[i] * 255);
      imageData.data[i * 4] = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }, [example]);

  const isCorrect = example.trueLabel === example.predictedLabel;

  return (
    <div className={`tr-card${isCorrect ? '' : ' tr-card-wrong'}`}>
      <canvas ref={canvasRef} className="tr-canvas" />
      <div className="tr-info">
        <div className="tr-row">
          <span className="tr-label">True:</span>
          <span className="tr-digit">{DIGITS[example.trueLabel]}</span>
        </div>
        <div className="tr-row">
          <span className="tr-label">Pred:</span>
          <span className={`tr-digit${isCorrect ? ' tr-digit-correct' : ' tr-digit-wrong'}`}>
            {DIGITS[example.predictedLabel]}
          </span>
        </div>
        <div className="tr-row">
          <span className="tr-label">Conf:</span>
          <span className="tr-conf">{(example.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="tr-badge">
          {isCorrect ? '✅ Correct' : '❌ Wrong'}
        </div>
      </div>
    </div>
  );
}

export function TestResults({ examples, accuracy }: TestResultsProps) {
  return (
    <div className="tr-container">
      <h3>Test Set Results</h3>
      <p className="tr-summary">
        Overall test accuracy: <strong className="tr-accuracy">{(accuracy * 100).toFixed(1)}%</strong>
        {' '}on {examples.length > 0 ? `a subset of ${examples.length} test images` : 'the test set'}.
      </p>

      <div className="tr-grid">
        {examples.map((ex, i) => (
          <ExampleCard key={i} example={ex} />
        ))}
      </div>
    </div>
  );
}
