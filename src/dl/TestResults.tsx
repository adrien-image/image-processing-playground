import { useEffect, useRef } from 'react';
import type { TestExample } from './lenet';
import { DIGITS } from './constants';

interface TestResultsProps {
  examples: TestExample[];
  accuracy: number;
  total: number;
}

/** 28×28 digit, drawn from 0–1 pixel values. */
export function DigitCanvas({ pixels, className, label }: { pixels: Float32Array; className: string; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(pixels[i] * 255);
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);
  return <canvas ref={canvasRef} width={28} height={28} className={className} role="img" aria-label={label} />;
}

function ExampleCard({ example }: { example: TestExample }) {
  const isCorrect = example.trueLabel === example.predictedLabel;
  return (
    <div className={`tr-card${isCorrect ? '' : ' tr-card-wrong'}`}>
      <DigitCanvas pixels={example.pixels} className="tr-canvas" label={`Test digit, true label ${example.trueLabel}`} />
      <div className="tr-info">
        <div className="tr-row">
          <span className="tr-label">True:</span>
          <span className="tr-digit">{DIGITS[example.trueLabel]}</span>
        </div>
        <div className="tr-row">
          <span className="tr-label">Pred:</span>
          <span className={`tr-digit${isCorrect ? ' tr-digit-correct' : ' tr-digit-wrong'}`}>{DIGITS[example.predictedLabel]}</span>
        </div>
        <div className="tr-row">
          <span className="tr-label">Conf:</span>
          <span className="tr-conf">{(example.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="tr-badge">{isCorrect ? '✅ Correct' : '❌ Wrong'}</div>
      </div>
    </div>
  );
}

export function TestResults({ examples, accuracy, total }: TestResultsProps) {
  const wrong = examples.filter(e => e.trueLabel !== e.predictedLabel).length;
  return (
    <div className="tr-container">
      <h3>Test Set Results</h3>
      <p className="tr-summary">
        Test accuracy: <strong className="tr-accuracy">{(accuracy * 100).toFixed(1)}%</strong> on all {total.toLocaleString()} held-out test images
        (never used for training or validation). Below: {examples.length - wrong} correct predictions, one per digit, and
        {wrong > 0 ? ` the ${wrong} most confident mistake${wrong > 1 ? 's' : ''}.` : ' no mistakes to show.'}
      </p>
      <div className="tr-grid">
        {examples.map((ex, i) => <ExampleCard key={i} example={ex} />)}
      </div>
    </div>
  );
}
