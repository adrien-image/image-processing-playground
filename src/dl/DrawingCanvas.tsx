import { useRef, useEffect, useCallback } from 'react';

interface DrawingCanvasProps {
  onCanvasReady: (canvas: HTMLCanvasElement) => void;
  /** Triggered when the canvas is cleared */
  onClear?: () => void;
}

const DRAW_SIZE = 280; // 10× the 28×28 MNIST input

export function DrawingCanvas({ onCanvasReady, onClear }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);

  useEffect(() => {
    if (canvasRef.current) onCanvasReady(canvasRef.current);
  }, [onCanvasReady]);

  const getPos = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    if ('touches' in e) {
      const touch = e.touches[0];
      return { x: (touch.clientX - rect.left) * (DRAW_SIZE / rect.width), y: (touch.clientY - rect.top) * (DRAW_SIZE / rect.height) };
    }
    return { x: (e.clientX - rect.left) * (DRAW_SIZE / rect.width), y: (e.clientY - rect.top) * (DRAW_SIZE / rect.height) };
  }, []);

  const startDraw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    isDrawing.current = true;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }, [getPos]);

  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!isDrawing.current) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const pos = getPos(e);
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#fff';
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }, [getPos]);

  const stopDraw = useCallback(() => {
    isDrawing.current = false;
  }, []);

  const clear = useCallback(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
    if (onClear) onClear();
  }, [onClear]);

  // Initialise canvas with black background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = DRAW_SIZE;
    canvas.height = DRAW_SIZE;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
  }, []);

  return (
    <div className="drawing-section">
      <h3>Draw a digit</h3>
      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        onMouseDown={startDraw}
        onMouseMove={draw}
        onMouseUp={stopDraw}
        onMouseLeave={stopDraw}
        onTouchStart={startDraw}
        onTouchMove={draw}
        onTouchEnd={stopDraw}
      />
      <div className="drawing-actions">
        <button className="btn btn-sm" onClick={clear}>Clear</button>
      </div>
    </div>
  );
}
