import { registerProcessor } from './registry';
import { SobelProcessor } from './sobel';
import { SobelGyProcessor } from './sobel/gy';
import { GaussianBlurProcessor } from './gaussian-blur';
import { SharpenProcessor } from './sharpen';

/**
 * Register all built-in processors.
 * Call once at app startup.
 */
export function initProcessors(): void {
  registerProcessor(SobelProcessor);
  registerProcessor(SobelGyProcessor);
  registerProcessor(GaussianBlurProcessor);
  registerProcessor(SharpenProcessor);
}
