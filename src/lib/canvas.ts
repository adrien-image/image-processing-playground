/** Browser-only canvas helpers shared by the sections. */

/** Create a canvas and its 2D context, optimised for repeated `getImageData` reads. */
export function createCanvas(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context is not available');
  return { canvas, ctx };
}

/** Scale (w, h) down so the longest side is at most `maxSize`; never scales up. */
export function fitWithin(width: number, height: number, maxSize: number): { width: number; height: number } {
  if (width <= maxSize && height <= maxSize) return { width, height };
  const ratio = Math.min(maxSize / width, maxSize / height);
  return { width: Math.max(1, Math.round(width * ratio)), height: Math.max(1, Math.round(height * ratio)) };
}

/**
 * Hide an ImageData's pixel buffer from property enumeration.
 *
 * Chromium stores `ImageData.data` as an own, enumerable property, and React's
 * development build diffs changed props for its performance timeline by enumerating
 * them — so every changed ImageData prop had each of its bytes listed (0.3–3 s per
 * slider move under `npm run dev`). Marking `data` non-enumerable keeps it fully
 * usable (`img.data`, `putImageData`) while the diff skips it. Firefox and Safari
 * already keep `data` on the prototype, where it is never enumerated.
 */
export function quietImageData<T extends { data: unknown }>(image: T): T {
  if (Object.prototype.hasOwnProperty.call(image, 'data')) {
    const desc = Object.getOwnPropertyDescriptor(image, 'data');
    if (desc?.configurable && desc.enumerable) Object.defineProperty(image, 'data', { enumerable: false });
  }
  return image;
}

/** Wrap an RGBA buffer in an ImageData. */
export function toImageData(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): ImageData {
  return quietImageData(new ImageData(rgba, width, height));
}

/** Copy an ImageData into a fresh canvas (e.g. to use it as a `drawImage` source). */
export function imageDataToCanvas(image: ImageData): HTMLCanvasElement {
  const { canvas, ctx } = createCanvas(image.width, image.height);
  ctx.putImageData(image, 0, 0);
  return canvas;
}

/** Read back the pixels of a canvas. */
export function canvasToImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context is not available');
  return quietImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
}
