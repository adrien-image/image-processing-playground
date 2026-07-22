import { useState, useCallback, useRef, useEffect } from 'react';
import { rgbaToGrayscale, grayscaleToRGBA, convolve } from '../processors/utils';

interface ProgressiveProcessingProps {
  imageData: ImageData | null;
  active: boolean;
  kernel: number[][];
  mode: 'absolute' | 'direct';
  onPartialResult: (imageData: ImageData) => void;
  onStep: (col: number, row: number) => void;
  onMathData?: (data: MathPanelData) => void;
}

export interface MathPanelData {
  neighbourhood: number[][];
  kernel: number[][];
  products: number[][];
  sum: number;
  output: number;
}

/**
 * Steps through every interior pixel, applying the given kernel via convolution.
 * Pixels are revealed one by one so the processed image builds up progressively.
 */
export function ProgressiveProcessing({
  imageData,
  active,
  kernel,
  mode,
  onPartialResult,
  onStep,
  onMathData,
}: ProgressiveProcessingProps) {
  const convRef = useRef<Float64Array | null>(null);
  const grayRef = useRef<Float64Array | null>(null);
  const orderRef = useRef<{ col: number; row: number }[]>([]);
  const outputRef = useRef<Float64Array | null>(null);
  const maxRef = useRef(0);
  const wRef = useRef(0);
  const hRef = useRef(0);

  const [stepOrder, setStepOrder] = useState<{ col: number; row: number }[]>([]);
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(100);
  const [mathData, setMathData] = useState<MathPanelData | null>(null);
  const timerRef = useRef<number | null>(null);

  // Notify parent whenever math data changes
  useEffect(() => {
    if (mathData && onMathData) onMathData(mathData);
  }, [mathData, onMathData]);

  /** Extract neighbourhood and compute products/sum/output for the current pixel. */
  const computeMathData = useCallback((col: number, row: number): MathPanelData | null => {
    const gray = grayRef.current;
    const w = wRef.current;
    const h = hRef.current;
    const maxV = maxRef.current;
    const k = kernel;

    if (!gray || !w || !h) return null;

    // Extract 3×3 neighbourhood from grayscale data
    const neighbourhood: number[][] = [];
    const products: number[][] = [];
    let sum = 0;

    for (let ky = -1; ky <= 1; ky++) {
      const nRow: number[] = [];
      const pRow: number[] = [];
      for (let kx = -1; kx <= 1; kx++) {
        const px = col + kx;
        const py = row + ky;
        const pixel = (px >= 0 && px < w && py >= 0 && py < h)
          ? gray[py * w + px]
          : 0;
        const kVal = k[ky + 1][kx + 1];
        nRow.push(Math.round(pixel));
        pRow.push(+(pixel * kVal).toFixed(1));
        sum += pixel * kVal;
      }
      neighbourhood.push(nRow);
      products.push(pRow);
    }

    // Compute final output (same formula as the main processing pipeline)
    const absVal = mode === 'absolute' ? Math.abs(sum) : sum;
    const scale = maxV > 0 ? 255 / maxV : 1;
    const output = Math.max(0, Math.min(255, Math.round(absVal * scale)));

    return { neighbourhood, kernel: k, products, sum: +sum.toFixed(1), output };
  }, [kernel, mode]);

  const emitPartial = useCallback((idx: number) => {
    const out = outputRef.current;
    const w = wRef.current;
    const h = hRef.current;
    if (!out || !w || !h) return;

    const rgba = grayscaleToRGBA(out, w, h);
    const result = new ImageData(rgba, w, h);
    onPartialResult(result);

    const step = orderRef.current[idx];
    if (step) {
      onStep(step.col, step.row);
      setMathData(computeMathData(step.col, step.row));
    }
  }, [onPartialResult, onStep, computeMathData]);

  // Precompute when image, kernel, or mode changes
  useEffect(() => {
    if (!imageData || !active) {
      setStepOrder([]);
      setMathData(null);
      setIsPlaying(false);
      return;
    }

    const { data, width: w, height: h } = imageData;
    const gray = rgbaToGrayscale(data, w, h);
    const raw = convolve(gray, w, h, kernel);

    // Apply mode and find max for normalisation
    const normalised = new Float64Array(w * h);
    let max = 0;
    for (let i = 0; i < w * h; i++) {
      const v = mode === 'absolute' ? Math.abs(raw[i]) : raw[i];
      normalised[i] = v;
      if (v > max) max = v;
    }

    // Snake-order pixel walk covering all interior pixels
    const order: { col: number; row: number }[] = [];
    for (let row = 1; row < h - 1; row++) {
      if (row % 2 === 1) {
        for (let col = 1; col < w - 1; col++) order.push({ col, row });
      } else {
        for (let col = w - 2; col >= 1; col--) order.push({ col, row });
      }
    }

    outputRef.current = new Float64Array(w * h);
    convRef.current = normalised;
    grayRef.current = gray;
    orderRef.current = order;
    maxRef.current = max;
    wRef.current = w;
    hRef.current = h;

    setStepOrder(order);
    setStepIndex(0);
    setMathData(null);
    emitPartial(0);
  }, [imageData, active, kernel, mode, emitPartial]);

  // Auto-play timer
  useEffect(() => {
    if (isPlaying && stepOrder.length > 0) {
      timerRef.current = window.setTimeout(() => {
        setStepIndex(prev => {
          if (prev >= orderRef.current.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    }
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying, stepIndex, stepOrder.length, speed]);

  // Advance the partial output whenever stepIndex changes
  useEffect(() => {
    const conv = convRef.current;
    const out = outputRef.current;
    const order = orderRef.current;
    const max = maxRef.current;
    const w = wRef.current;
    const h = hRef.current;

    if (!conv || !out || order.length === 0 || !w || !h) return;

    const idx = stepIndex;
    const step = order[idx];
    if (!step) return;

    const scale = max > 0 ? 255 / max : 1;
    out[step.row * w + step.col] = conv[step.row * w + step.col] * scale;
    emitPartial(idx);
  }, [stepIndex, emitPartial]);

  const rebuildOutput = useCallback((targetIdx: number) => {
    const conv = convRef.current;
    const order = orderRef.current;
    const max = maxRef.current;
    const w = wRef.current;
    const h = hRef.current;

    if (!conv || order.length === 0 || !w || !h) return;

    const scale = max > 0 ? 255 / max : 1;
    const out = new Float64Array(w * h);
    for (let i = 0; i <= targetIdx && i < order.length; i++) {
      const { col, row } = order[i];
      out[row * w + col] = conv[row * w + col] * scale;
    }
    outputRef.current = out;
    setStepIndex(targetIdx);

    const rgba = grayscaleToRGBA(out, w, h);
    const result = new ImageData(rgba, w, h);
    onPartialResult(result);

    const step = order[targetIdx];
    if (step) {
      onStep(step.col, step.row);
      setMathData(computeMathData(step.col, step.row));
    }
  }, [onPartialResult, onStep, computeMathData]);

  const goTo = useCallback((idx: number) => {
    const clamped = Math.max(0, Math.min(idx, orderRef.current.length - 1));
    rebuildOutput(clamped);
  }, [rebuildOutput]);

  const togglePlay = useCallback(() => {
    if (stepIndex >= orderRef.current.length - 1) goTo(0);
    setIsPlaying(p => !p);
  }, [stepIndex, goTo]);

  const step = orderRef.current[stepIndex];
  const total = orderRef.current.length;
  const hasData = stepOrder.length > 0;

  return (
    <div className="progressive-processing">
      <div className="proc-controls">
        <div className="proc-controls-left">
          <button className="btn btn-sm" onClick={() => goTo(0)} disabled={stepIndex === 0}>⏮</button>
          <button className="btn btn-sm" onClick={() => goTo(stepIndex - 1)} disabled={stepIndex === 0}>◀</button>
          <button className="btn btn-sm" onClick={togglePlay}>{isPlaying ? '⏸' : '▶'}</button>
          <button className="btn btn-sm" onClick={() => goTo(stepIndex + 1)} disabled={stepIndex >= total - 1}>▶</button>
          <button className="btn btn-sm" onClick={() => goTo(total - 1)} disabled={stepIndex >= total - 1}>⏭</button>
        </div>

        <div className="proc-info">
          {step ? <span className="proc-pixel">Pixel ({step.col}, {step.row})</span> : <span className="proc-pixel">—</span>}
          <span className="proc-progress">{hasData ? `${Math.min(stepIndex + 1, total)} / ${total}` : '—'}</span>
        </div>

        <div className="proc-controls-right">
          <div className="speed-control">
            <label>Speed</label>
            <input type="range" min={10} max={1000} step={10} value={1010 - speed}
              onChange={(e) => setSpeed(1010 - Number(e.target.value))} />
          </div>
        </div>
      </div>

      <div className="proc-progress-bar">
        <input type="range" min={0} max={Math.max(total - 1, 0)} value={stepIndex}
          onChange={(e) => goTo(Number(e.target.value))} />
      </div>

      {/* Math panel */}
      {mathData && hasData && (
        <MathPanel data={mathData} />
      )}
    </div>
  );
}

/** Displays the convolution computation for the current pixel. */
export function MathPanel({ data }: { data: MathPanelData }) {
  const { neighbourhood, kernel, products, sum, output } = data;

  // Build the arithmetic expression: (n1 × k1) + (n2 × k2) + ... = sum
  const flatNb = neighbourhood.flat();
  const flatK = kernel.flat();
  const terms = flatNb.map((n, i) => `(${n} × ${flatK[i]})`);
  const expression = terms.join(' + ') + ` = ${sum}`;

  return (
    <div className="math-panel">
      <div className="math-panel-title">Convolution computation</div>
      <div className="math-panel-row">
        {/* Neighbourhood */}
        <div className="math-block">
          <div className="math-block-label">Neighbourhood</div>
          <div className="math-grid">
            {flatNb.map((v, i) => (
              <span key={i} className="math-cell math-cell-nb">{v}</span>
            ))}
          </div>
        </div>

        <span className="math-op">×</span>

        {/* Kernel */}
        <div className="math-block">
          <div className="math-block-label">Kernel</div>
          <div className="math-grid">
            {flatK.map((v, i) => (
              <span key={i} className="math-cell math-cell-kernel">{v}</span>
            ))}
          </div>
        </div>

        <span className="math-op">=</span>

        {/* Products */}
        <div className="math-block">
          <div className="math-block-label">Products</div>
          <div className="math-grid">
            {products.flat().map((v, i) => (
              <span key={i} className="math-cell math-cell-prod">{v}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Arithmetic expression */}
      <div className="math-expression">{expression}</div>

      {/* Sum + Output */}
      <div className="math-result">
        <div className="math-result-item">
          <span className="math-result-label">Σ (sum)</span>
          <span className="math-result-value">{sum}</span>
        </div>
        <div className="math-result-item">
          <span className="math-result-label">Output</span>
          <span className="math-result-value math-result-output">{output}</span>
        </div>
      </div>
    </div>
  );
}
