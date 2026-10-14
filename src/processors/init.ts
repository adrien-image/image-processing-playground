import { registerProcessor } from './registry';
import { SobelXProcessor, SobelYProcessor } from './sobel';
import { GaussianBlurProcessor } from './gaussian-blur';
import { SharpenProcessor } from './sharpen';

/**
 * Register all built-in processors. Called once from `main.tsx`, before the first
 * render, so the registry is populated by the time any component reads it.
 * Safe to call more than once.
 */
export function initProcessors(): void {
  registerProcessor(SobelXProcessor);
  registerProcessor(SobelYProcessor);
  registerProcessor(GaussianBlurProcessor);
  registerProcessor(SharpenProcessor);
}
