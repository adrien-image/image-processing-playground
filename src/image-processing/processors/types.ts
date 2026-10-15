import type { OutputMode } from './utils';

export type ParamValue = number | boolean | string;

/** User-adjusted parameter values, keyed by param name. */
export type ProcessorConfig = Record<string, ParamValue>;

/** Parameter types supported by processor configuration UIs */
export type ParamType = 'number' | 'boolean' | 'select';

/** Description of a single configurable parameter for a processor */
export interface ParamDefinition {
  type: ParamType;
  label: string;
  default: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  options?: { label: string; value: string }[]; // for 'select' type
}

/**
 * A convolution filter that the Convolution section can animate pixel by pixel.
 * The UI discovers registered processors and renders controls from `params`.
 */
export interface Processor {
  /** Unique identifier (e.g. "sobel-x") */
  id: string;
  /** Human-readable name (e.g. "Sobel Gx — Vertical Edges") */
  name: string;
  /** Short description shown in the processor selector */
  description: string;
  /** Parameter schema — the UI auto-renders controls from this */
  params: Record<string, ParamDefinition>;
  /**
   * The kernel to slide over the image for the given parameter values.
   * Must be square with an odd size (3×3, 5×5, …).
   */
  kernel(config: ProcessorConfig): number[][];
  /** How raw sums become 0–255 output values (see `OutputMode`). */
  output: OutputMode;
}
