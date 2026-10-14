import type { Processor, ProcessorConfig } from './types';
import { convolve, makeOutputMapper } from './utils';

/** Parameter values a processor starts with. */
export function defaultConfig(processor: Processor): ProcessorConfig {
  const config: ProcessorConfig = {};
  for (const [key, def] of Object.entries(processor.params)) config[key] = def.default;
  return config;
}

/** Fill in any parameters missing from `config` with their defaults. */
export function resolveConfig(processor: Processor, config: ProcessorConfig): ProcessorConfig {
  return { ...defaultConfig(processor), ...config };
}

/**
 * Run a processor over a grayscale image (values 0–255).
 * Returns the raw convolution sums and the final 0–255 output.
 */
export function applyProcessor(
  processor: Processor,
  gray: Float64Array,
  width: number,
  height: number,
  config: ProcessorConfig = {},
): { raw: Float64Array; output: Float64Array; kernel: number[][] } {
  const kernel = processor.kernel(resolveConfig(processor, config));
  const raw = convolve(gray, width, height, kernel);
  const map = makeOutputMapper(raw, processor.output);
  const output = new Float64Array(raw.length);
  for (let i = 0; i < raw.length; i++) output[i] = map(raw[i]);
  return { raw, output, kernel };
}
