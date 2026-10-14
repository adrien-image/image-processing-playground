import { grayscaleToRGBA } from '../processors/utils';
import { toImageData } from './canvas';

/** A single-channel image with values in 0–255. */
export interface GrayImage {
  pixels: Float64Array;
  width: number;
  height: number;
}

/** Render a gray image (values clamped to 0–255) as displayable ImageData. */
export function grayToImageData(gray: GrayImage): ImageData;
export function grayToImageData(pixels: Float64Array, width: number, height: number): ImageData;
export function grayToImageData(a: GrayImage | Float64Array, w?: number, h?: number): ImageData {
  if (a instanceof Float64Array) return toImageData(grayscaleToRGBA(a, w!, h!), w!, h!);
  return toImageData(grayscaleToRGBA(a.pixels, a.width, a.height), a.width, a.height);
}

/** Build ImageData from a per-pixel colour function. */
export function rgbImage(width: number, height: number, colour: (i: number) => readonly [number, number, number]): ImageData {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const [r, g, b] = colour(i);
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
  }
  return toImageData(rgba, width, height);
}
