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

/** Wrap an RGBA buffer in an ImageData. */
export function toImageData(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): ImageData {
  return new ImageData(rgba, width, height);
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
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
