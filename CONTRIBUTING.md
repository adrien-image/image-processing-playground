# Contributing

## Before you push

```bash
npm run lint && npm run typecheck && npm test && npm run build
```

CI runs the same four checks on every pull request.

- TypeScript is in `strict` mode.
- Algorithms live in pure `.ts` modules next to their section and come with a `*.test.ts`. Keep DOM and React out of them, so they run under Vitest in Node.
- UI state flows one way: section state → `useMemo` derived pixels → `<PixelCanvas image={…} />`. Load images with `useImageData(source, maxSize)`, which also downscales uploads.

## Adding a new convolution filter

The Convolution section uses a **Processor Registry**. A processor describes a kernel; the section
animates it, shows its math and renders its parameter controls. You don't touch any UI code.

### 1. Create the processor

```ts
// src/processors/box-blur/index.ts
import type { Processor } from '../types';

export const BoxBlurProcessor: Processor = {
  id: 'box-blur',
  name: 'Box Blur',
  description: 'Replaces each pixel with the plain average of its neighbourhood.',

  params: {
    // Rendered as a slider; the value arrives in `config.weight`.
    weight: { type: 'number', label: 'Centre weight', default: 1, min: 1, max: 8, step: 1 },
  },

  // Return the kernel for the current parameter values (square, odd size).
  kernel(config) {
    const c = Number(config.weight);
    const total = 8 + c;
    return [
      [1 / total, 1 / total, 1 / total],
      [1 / total, c / total, 1 / total],
      [1 / total, 1 / total, 1 / total],
    ];
  },

  // 'clamp' for kernels whose weights sum to 1 (blur, sharpen);
  // 'magnitude' for signed edge detectors (|sum| scaled so the strongest response is 255).
  output: 'clamp',
};
```

Reusable kernels (Sobel, Prewitt, Laplacian, `gaussianKernel`, `sharpenKernel`) are in
`src/processors/kernels.ts`.

### 2. Register it

```ts
// src/processors/init.ts
import { BoxBlurProcessor } from './box-blur';

export function initProcessors(): void {
  // …existing processors…
  registerProcessor(BoxBlurProcessor);
}
```

### 3. Done

The processor now appears in the Convolution section's list, with its kernel, parameter
controls, step-by-step animation and math panel. `processors.test.ts` automatically checks that
every registered processor returns an odd, square kernel. Add a test of your own for its output.

## Processor interface

```ts
interface Processor {
  id: string;                               // unique key, e.g. "box-blur"
  name: string;                             // shown in the selector
  description: string;                      // shown under the selector
  params: Record<string, ParamDefinition>;  // schema → auto-generated controls
  kernel(config: ProcessorConfig): number[][]; // odd, square kernel
  output: 'magnitude' | 'clamp';            // how raw sums become 0–255
}
```

To run a processor outside the UI, use `applyProcessor(processor, gray, width, height, config)` from
`src/processors`. It returns `{ raw, output, kernel }`.

### Supported param types

| Type      | Widget       | Extra fields                     |
|-----------|--------------|----------------------------------|
| `number`  | Range slider | `min`, `max`, `step`             |
| `boolean` | Checkbox     | —                                |
| `select`  | Dropdown     | `options: [{label, value}, ...]` |

## Utility functions

In `src/processors/utils.ts`:

- **`convolve(pixels, width, height, kernel)`** — 2-D sliding-window sum with replicate padding (strictly, cross-correlation: the kernel isn't flipped).
- **`makeOutputMapper(raw, mode)`** — maps raw sums to 0–255 for an `OutputMode`.
- **`rgbaToGrayscale(data, width, height)`** / **`grayscaleToRGBA(pixels, width, height)`** — RGBA ↔ single-channel `Float64Array`.
- **`normaliseMagnitude(data)`** / **`normaliseRange(data)`** — scale values for display.
- **`sampleClamped(pixels, width, height, x, y)`** — read a pixel with replicate padding.
