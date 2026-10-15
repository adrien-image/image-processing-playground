import { useEffect, useRef, type ReactNode } from 'react';
import './PixelCanvas.css';

interface PixelCanvasProps {
  /** Pixels to show; `null` renders the placeholder at the same size. */
  image: ImageData | null;
  /** Longest side in CSS pixels. The other side follows the image's aspect ratio. */
  size: number;
  /** Accessible description of what the image shows. */
  label: string;
  /** Nearest-neighbour upscaling, so individual pixels stay visible (default true). */
  pixelated?: boolean;
  /** Aspect ratio (w/h) to use while `image` is null. */
  placeholderAspect?: number;
  placeholder?: ReactNode;
  className?: string;
  /** Overlays, positioned relative to the image box (use percentages to stay responsive). */
  children?: ReactNode;
}

/**
 * Displays ImageData at a fixed CSS size while preserving its aspect ratio, so
 * non-square uploads are never stretched. Shrinks with the viewport on small screens.
 */
export function PixelCanvas({
  image,
  size,
  label,
  pixelated = true,
  placeholderAspect = 1,
  placeholder,
  className,
  children,
}: PixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.getContext('2d')?.putImageData(image, 0, 0);
  }, [image]);

  const aspect = image ? image.width / image.height : placeholderAspect;
  // Not rounded: for very thin portrait images rounding the width to whole pixels
  // would change the displayed height (via aspect-ratio), so panels showing the same
  // image at different resolutions would no longer line up.
  const width = aspect >= 1 ? size : size * aspect;

  return (
    <div
      className={`pixel-canvas${className ? ` ${className}` : ''}`}
      style={{ width, aspectRatio: `${aspect}` }}
    >
      {image ? (
        <canvas
          ref={canvasRef}
          className={`pixel-canvas-el${pixelated ? ' pixel-canvas-pixelated' : ''}`}
          role="img"
          aria-label={label}
        />
      ) : (
        <div className="pixel-canvas-placeholder">{placeholder}</div>
      )}
      {image && children}
    </div>
  );
}
