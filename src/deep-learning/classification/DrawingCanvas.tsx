import { useEffect, useRef, type PointerEvent } from 'react';

interface DrawingCanvasProps {
  onCanvasReady: (canvas: HTMLCanvasElement) => void;
  /** Called after each stroke ends */
  onStrokeEnd?: () => void;
  /** Triggered when the canvas is cleared */
  onClear?: () => void;
}

export const DRAW_SIZE = 280; // 10× the 28×28 MNIST input

/** Black canvas to draw a white digit on, with mouse, pen or touch (pointer events). */
export function DrawingCanvas({ onCanvasReady, onStrokeEnd, onClear }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = DRAW_SIZE;
    canvas.height = DRAW_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (ctx) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
    }
    onCanvasReady(canvas);
  }, [onCanvasReady]);

  const pos = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * DRAW_SIZE,
      y: ((e.clientY - rect.top) / rect.height) * DRAW_SIZE,
    };
  };

  const stroke = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };

  const start = (e: PointerEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    last.current = pos(e);
    stroke(last.current, last.current); // a dot for a single tap
  };

  const move = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current || !last.current) return;
    const p = pos(e);
    stroke(last.current, p);
    last.current = p;
  };

  const end = () => {
    if (!drawing.current) return;
    drawing.current = false;
    last.current = null;
    onStrokeEnd?.();
  };

  const clear = () => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
    }
    onClear?.();
  };

  return (
    <div className="drawing-section">
      <h3>Draw a digit</h3>
      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        role="img"
        aria-label="Drawing area: draw a digit with the mouse, a pen or your finger"
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
      />
      <div className="drawing-actions">
        <button type="button" className="btn btn-sm" onClick={clear}>Clear</button>
      </div>
    </div>
  );
}
