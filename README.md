# 🧠 Image Processing Playground

An interactive web app that teaches how image processing works — every algorithm runs in your browser,
on demo images or your own uploads, and every section can show you its source code.

## Sections

| Section | What you can explore |
|---|---|
| Convolution | See Sobel, Gaussian blur (3×3 to 9×9) and sharpen on a 256 px image, or switch to 32 px and step through the kernel pixel by pixel. Click any pixel to see its multiply-and-sum |
| Segmentation | Otsu thresholding vs. k-means clustering, with the histogram |
| Matching | Harris corners → rotation-normalised descriptors → RANSAC similarity estimate, compared with the ground truth |
| Color | RGB / HSV channels, hue, saturation, brightness and gamma adjustments |
| Morphology | Erosion, dilation, opening and closing on binary and grayscale images |
| Regions | Connected-component labelling (4/8-connectivity) and per-object measurements: area, centroid, bounding box, perimeter, circularity |
| Edges | Sobel, Prewitt, Laplacian and Canny side by side |
| Frequency | 2-D FFT spectrum, ideal and Gaussian low/high/band filters |
| Pyramids | Resampling and aliasing, nearest/bilinear/bicubic interpolation, Gaussian and Laplacian pyramids, multiband blending |
| Hough | Line segments and circles from the Hough transform, with the accumulators |
| Histogram | Histogram equalisation and its mapping curve |
| Augmentation | Chained random transformations used to grow training sets |
| Denoising | Median vs. Gaussian filtering on salt-and-pepper or Gaussian noise, scored with PSNR |
| Template | Normalised cross-correlation template matching with its score map |
| Classification | Train LeNet-5 on MNIST in the browser (TensorFlow.js), draw digits to classify, and look inside the network: learned filters, activation maps and occlusion sensitivity |

The home page (`#/home`) introduces the playground and lists every lesson. Each section is its own
lazily-loaded chunk, and the current section is kept in the URL (`#/edges`), so sections can be
bookmarked and shared.

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173/image-processing-playground/
```

| Script | |
|---|---|
| `npm run dev` | Dev server (also prepares the MNIST data) |
| `npm run build` | Type-check, then production build into `dist/` |
| `npm run preview` | Serve the production build |
| `npm test` | Unit tests (Vitest) |
| `npm run lint` | oxlint, warnings are errors |
| `npm run typecheck` | `tsc -b` |
| `npm run prepare-data` | Pack the MNIST subset into `public/data/` (runs automatically before `dev` and `build`) |

CI (`.github/workflows/deploy.yml`) runs lint, type-check, tests and build on every pull request
and push, and deploys `main` to GitHub Pages.

## Architecture

```
src/
├── main.tsx                 ← entry: fonts, global styles, processor registration
├── App.tsx                  ← header, lesson menu, hash routing, lazy sections
├── app/                     ← home page, site title/tagline, section list, hash router, error boundary
├── processors/              ← convolution filters for the Convolution section
│   ├── types.ts             ← Processor interface (the contract)
│   ├── registry.ts          ← central registry (the UI lists what is registered)
│   ├── init.ts              ← registers the built-in processors
│   ├── kernels.ts           ← every shared kernel (Sobel, Prewitt, Laplacian, Gaussian…)
│   └── utils.ts             ← convolve (replicate padding), output mapping, gray ↔ RGBA
├── hooks/useImageData.ts    ← load a URL or canvas → downscaled RGBA + grayscale pixels
├── components/              ← PixelCanvas, ImageUploader, CodeViewer (+ tokenizer)
├── lib/                     ← canvas helpers, demo images, seeded random
├── styles/base.css          ← design tokens and shared building blocks
└── <section>/               ← one folder per section:
    ├── <algorithm>.ts       ←   pure algorithm code (what "Show code" displays)
    ├── <algorithm>.test.ts  ←   its unit tests
    ├── <Section>.tsx        ←   UI
    └── <section>.css        ←   section styles
```

Algorithms are plain TypeScript functions over `Float64Array` pixels with no DOM access, so they are
unit-tested in Node. The UI derives everything from state with `useMemo`, and images are drawn
through one `PixelCanvas` component that keeps the aspect ratio.

`scripts/build-mnist-sprite.mjs` packs a balanced MNIST subset (600 train / 100 validation /
100 test images per digit) from the `mnist` package into a 1.2 MB gzip file, so the browser never
downloads the package's 20 MB of JSON.

## Adding a new convolution filter

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). In short: create `src/processors/your-filter/index.ts`
exporting a `Processor` (kernel, output mode, parameters), register it in `src/processors/init.ts`,
and it appears in the Convolution section with controls generated from its parameters.

## Tech stack

React 19 · TypeScript (strict) · Vite · Vitest · oxlint · TensorFlow.js
