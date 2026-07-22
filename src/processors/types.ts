/** Parameter types supported by processor configuration UIs */
export type ParamType = 'number' | 'boolean' | 'select';

/** Description of a single configurable parameter for a processor */
export interface ParamDefinition {
  type: ParamType;
  label: string;
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  options?: { label: string; value: string }[]; // for 'select' type
}

/** The interface every image processor must implement */
export interface Processor {
  /** Unique identifier (e.g. "sobel") */
  id: string;
  /** Human-readable name (e.g. "Sobel Edge Detection") */
  name: string;
  /** Short description shown in the processor selector */
  description: string;
  /** Parameter schema — the UI auto-renders controls from this */
  params: Record<string, ParamDefinition>;
  /**
   * Apply the processor to an image.
   * @param imageData - The source image pixel data
   * @param config - User-adjusted parameter values keyed by param name
   * @returns A new ImageData with the processed result
   */
  apply(imageData: ImageData, config: Record<string, number | boolean | string>): ImageData;
}

/** Step data for animated convolution visualisation */
export interface ConvolutionStep {
  /** Pixel coordinates (col, row) of the centre pixel */
  col: number;
  row: number;
  /** The 3×3 neighbourhood values (normalised 0–1) */
  neighbourhood: number[][];
  /** The kernel values */
  kernel: number[][];
  /** Element-wise products (neighbourhood × kernel) */
  products: number[][];
  /** Sum of all products — the raw convolution result */
  sum: number;
  /** Clamped output value (0–255) */
  output: number;
}
