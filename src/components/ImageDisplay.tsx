import { useRef, useEffect } from 'react';

interface ImageDisplayProps {
  original: ImageData | null;
  processed: ImageData | null;
  width: number;
  height: number;
  /** Optional pixel position to highlight on the original image (for animation) */
  highlightPosition: { col: number; row: number } | null;
}

export function ImageDisplay({ original, processed, width, height, highlightPosition }: ImageDisplayProps) {
  const origRef = useRef<HTMLCanvasElement>(null);
  const procRef = useRef<HTMLCanvasElement>(null);

  // Draw original image
  useEffect(() => {
    const canvas = origRef.current;
    if (!canvas || !original) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(original, 0, 0);
  }, [original, width, height]);

  // Draw processed image
  useEffect(() => {
    const canvas = procRef.current;
    if (!canvas || !processed) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(processed, 0, 0);
  }, [processed, width, height]);

  // Draw 3×3 convolution window overlay on the original image
  useEffect(() => {
    const canvas = origRef.current;
    if (!canvas || !original || !highlightPosition) return;
    const ctx = canvas.getContext('2d')!;

    // Re-draw the original first (to clear previous overlay)
    ctx.putImageData(original, 0, 0);

    const { col, row } = highlightPosition;

    // The 3×3 block spans pixels (col-1, row-1) → (col+1, row+1)
    const x0 = col - 1;
    const y0 = row - 1;

    ctx.save();

    // Translucent cyan fill for the 3×3 block
    ctx.fillStyle = 'rgba(0, 240, 255, 0.20)';
    ctx.fillRect(x0, y0, 3, 3);

    // Brighter centre pixel
    ctx.fillStyle = 'rgba(0, 240, 255, 0.40)';
    ctx.fillRect(col, row, 1, 1);

    // Grid lines — draw cell borders within the 3×3 block
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 3; i++) {
      ctx.beginPath();
      ctx.moveTo(x0 + i, y0);
      ctx.lineTo(x0 + i, y0 + 3);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x0, y0 + i);
      ctx.lineTo(x0 + 3, y0 + i);
      ctx.stroke();
    }

    // Outer glow border
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 8;
    ctx.strokeRect(x0, y0, 3, 3);

    // Centre-pixel crosshair (yellow)
    ctx.strokeStyle = '#ffdd44';
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(col + 0.5, row + 0.5 - 0.8);
    ctx.lineTo(col + 0.5, row + 0.5 + 0.8);
    ctx.moveTo(col + 0.5 - 0.8, row + 0.5);
    ctx.lineTo(col + 0.5 + 0.8, row + 0.5);
    ctx.stroke();

    ctx.restore();
  }, [original, highlightPosition]);

  // Scale so the longest side fits targetSize (magnifies small images,
  // shrinks large ones — the pixelated render makes individual pixels visible).
  const targetSize = 400;
  const scale = targetSize / Math.max(width, height, 1);
  const displayW = Math.round(width * scale);
  const displayH = Math.round(height * scale);

  return (
    <div className="image-display">
      <div className="image-panel">
        <h3>Original</h3>
        <div className="canvas-wrapper">
          <canvas
            ref={origRef}
            style={{
              width: displayW,
              height: displayH,
              imageRendering: 'pixelated',
            }}
            className="canvas"
          />
          {highlightPosition && (
            <div className="position-badge">
              ({highlightPosition.col}, {highlightPosition.row})
            </div>
          )}
        </div>
        <span className="dimensions">{width}×{height}</span>
      </div>

      <div className="image-panel">
        <h3>Processed</h3>
        <div className="canvas-wrapper">
          {processed ? (
            <canvas
              ref={procRef}
              style={{
                width: displayW,
                height: displayH,
                imageRendering: 'pixelated',
              }}
              className="canvas"
            />
          ) : (
            <div
              className="canvas placeholder"
              style={{ width: displayW, height: displayH }}
            >
              <span>Select a processor</span>
            </div>
          )}
        </div>
        <span className="dimensions">{width}×{height}</span>
      </div>
    </div>
  );
}
