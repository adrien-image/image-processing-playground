import { useEffect, useMemo, useState } from 'react';
import { rgbaToGrayscale } from '../processors/utils';
import { createCanvas, fitWithin, quietImageData } from '../lib/canvas';

/** A URL (data:, blob:, same-origin) or an already-drawn canvas. */
export type ImageSource = string | HTMLCanvasElement;

export interface LoadedImage {
  /** RGBA pixels, downscaled so the longest side is at most `maxSize`. */
  imageData: ImageData;
  /** Luma (0–255) of the same pixels. */
  gray: Float64Array;
  width: number;
  height: number;
}

/** Refuse to decode absurdly large images (protects the tab's memory). */
const MAX_SOURCE_PIXELS = 50_000_000;

function rasterise(source: CanvasImageSource, srcW: number, srcH: number, maxSize: number): LoadedImage {
  if (srcW * srcH > MAX_SOURCE_PIXELS) {
    throw new Error(`Image too large (${srcW}×${srcH}); the limit is ${MAX_SOURCE_PIXELS / 1e6} megapixels.`);
  }
  const { width, height } = fitWithin(srcW, srcH, maxSize);
  const { ctx } = createCanvas(width, height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, width, height);
  const imageData = quietImageData(ctx.getImageData(0, 0, width, height));
  return { imageData, gray: rgbaToGrayscale(imageData.data, width, height), width, height };
}

interface UrlState {
  url: string | null;
  maxSize: number;
  image: LoadedImage | null;
  error: string | null;
}

/**
 * Load an image source into pixel arrays, downscaled to fit `maxSize`.
 *
 * - Canvas sources are rasterised synchronously.
 * - URL sources load asynchronously; while a new URL loads, the previous image is kept
 *   (`loading` is true) and results for stale URLs are discarded.
 * - Load/decode failures surface as `error` instead of hanging.
 *
 * Pass a stable `source` (state or memoised) — a new canvas object means a new image.
 */
export function useImageData(source: ImageSource | null, maxSize: number): {
  image: LoadedImage | null;
  error: string | null;
  loading: boolean;
} {
  const canvasResult = useMemo(() => {
    if (!(source instanceof HTMLCanvasElement)) return null;
    try {
      return { image: rasterise(source, source.width, source.height, maxSize), error: null };
    } catch (e) {
      return { image: null, error: e instanceof Error ? e.message : 'Failed to read image' };
    }
  }, [source, maxSize]);

  const url = typeof source === 'string' ? source : null;
  const [urlState, setUrlState] = useState<UrlState>({ url: null, maxSize, image: null, error: null });

  useEffect(() => {
    if (url === null) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      try {
        const image = rasterise(img, img.naturalWidth, img.naturalHeight, maxSize);
        setUrlState({ url, maxSize, image, error: null });
      } catch (e) {
        setUrlState({ url, maxSize, image: null, error: e instanceof Error ? e.message : 'Failed to read image' });
      }
    };
    img.onerror = () => {
      if (!cancelled) setUrlState({ url, maxSize, image: null, error: 'Could not decode this file as an image.' });
    };
    img.src = url;
    return () => {
      cancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [url, maxSize]);

  if (canvasResult) return { ...canvasResult, loading: false };
  if (url === null) return { image: null, error: null, loading: false };
  const current = urlState.url === url && urlState.maxSize === maxSize;
  return { image: urlState.image, error: current ? urlState.error : null, loading: !current };
}
