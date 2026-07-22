# Contributing — Adding a New Image Processor

This project uses a **Processor Registry** pattern. Adding a new filter requires **zero changes** to existing UI code.

## Architecture

```
src/processors/
├── types.ts          ← Processor interface (the contract)
├── registry.ts       ← Central registry (auto-discovered by UI)
├── init.ts           ← Registration bootstrap
├── utils.ts          ← Shared convolution math helpers
├── sobel/            ← Example processor
│   ├── kernels.ts
│   └── index.ts
└── your-filter/      ← YOUR NEW PROCESSOR
    └── index.ts
```

The UI queries `getAllProcessors()` from the registry and auto-renders:
- A selector entry (from `name` + `description`)
- Parameter controls (from `params` schema)

## Step-by-Step: Add a Gaussian Blur

### 1. Create the processor file

```ts
// src/processors/gaussian-blur/index.ts
import type { Processor } from '../types';
import { convolve, rgbaToGrayscale, grayscaleToRGBA } from '../utils';

const GAUSSIAN_3x3: number[][] = [
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1],
];

export const GaussianBlurProcessor: Processor = {
  id: 'gaussian-blur',
  name: 'Gaussian Blur',
  description: 'Smooths the image using a 3×3 Gaussian kernel.',

  params: {
    strength: {
      type: 'number',
      label: 'Strength',
      default: 1,
      min: 0,
      max: 5,
      step: 1,
    },
  },

  apply(imageData, config) {
    const { data, width, height } = imageData;
    const strength = (config.strength as number) ?? 1;

    // Normalise kernel
    const total = GAUSSIAN_3x3.flat().reduce((a, b) => a + b, 0);
    const kernel = GAUSSIAN_3x3.map(row =>
      row.map(v => (v / total) * strength)
    );

    const gray = rgbaToGrayscale(data, width, height);
    const result = convolve(gray, width, height, kernel);
    return new ImageData(
      grayscaleToRGBA(result, width, height),
      width,
      height,
    );
  },
};
```

### 2. Register it

```ts
// src/processors/init.ts
import { GaussianBlurProcessor } from './gaussian-blur';

export function initProcessors(): void {
  registerProcessor(SobelProcessor);
  registerProcessor(GaussianBlurProcessor);    // ← add this line
}
```

### 3. Done!

The UI automatically discovers the new processor on next reload.

## Processor Interface Reference

```ts
interface Processor {
  id: string;                             // Unique key (e.g. "gaussian-blur")
  name: string;                           // Human-readable name
  description: string;                    // Shown in selector
  params: Record<string, ParamDefinition>; // Schema → auto-generated controls
  apply(data: ImageData, config: Record<string, number | boolean | string>): ImageData;
}
```

### Supported param types

| Type      | Widget       | Extra fields                     |
|-----------|--------------|----------------------------------|
| `number`  | Range slider | `min`, `max`, `step`             |
| `boolean` | Checkbox     | —                                |
| `select`  | Dropdown     | `options: [{label, value}, ...]` |

## Utility Functions

All in `src/processors/utils.ts`:

- **`convolve(pixels, width, height, kernel)`** — 2D convolution, zero-padded borders
- **`rgbaToGrayscale(data, width, height)`** — RGBA → single-channel Float64Array
- **`grayscaleToRGBA(pixels, width, height)`** — Float64Array → RGBA Uint8ClampedArray
- **`generateConvolutionSteps(pixels, width, height, kernel, col, row, count)`** — animation data
