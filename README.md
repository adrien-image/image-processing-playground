# 🧠 Image Processing Playground

An interactive web app that teaches how image processing works — every algorithm runs in your browser,
on demo images or your own uploads, and every section can show you its source code.

## Lessons

The lessons come in two groups, which are also the two source folders: `src/image-processing/` and
`src/deep-learning/`.

### Image processing

| Lesson | What you can explore |
|---|---|
| Convolution | See Sobel, Gaussian blur (3×3 to 9×9) and sharpen on a 256 px image, or switch to 32 px and step through the kernel pixel by pixel. Click any pixel to see its multiply-and-sum |
| Segmentation | Otsu thresholding vs. k-means clustering, with the histogram |
| Superpixels | SLIC step by step: k-means on colour and position, compactness, mean-colour regions |
| Matching | Harris corners → rotation-normalised descriptors → RANSAC similarity estimate, compared with the ground truth |
| Color | RGB / HSV channels, hue, saturation, brightness and gamma adjustments |
| Morphology | Erosion, dilation, opening and closing on binary and grayscale images |
| Regions | Connected-component labelling (4/8-connectivity) and per-object measurements: area, centroid, bounding box, perimeter, circularity |
| Watershed | Touching coins split with the distance transform, markers and an animated flood |
| Edges | Sobel, Prewitt, Laplacian and Canny side by side |
| Frequency | 2-D FFT spectrum, ideal and Gaussian low/high/band filters |
| JPEG | 8×8 DCT, quality-scaled quantisation, zigzag and an entropy estimate of the file size; a block inspector and the 64 basis images |
| Pyramids | Resampling and aliasing, nearest/bilinear/bicubic interpolation, Gaussian and Laplacian pyramids, multiband blending |
| Hough | Line segments and circles from the Hough transform, with the accumulators |
| Histogram | Histogram equalisation and its mapping curve |
| Augmentation | Chained random transformations used to grow training sets |
| Denoising | Median vs. Gaussian filtering on salt-and-pepper or Gaussian noise, scored with PSNR |
| Deblurring | Motion, Gaussian and defocus blur; inverse filter vs. Wiener vs. Richardson–Lucy, with the blur's frequency response |
| Inpainting | Paint a hole and fill it by diffusion or by copying patches (Criminisi), patch by patch |
| Template | Normalised cross-correlation template matching with its score map |
| Optical flow | Pyramidal Lucas–Kanade on synthetic motions, the aperture problem, errors against the true flow |
| Stereo depth | Block matching along rows, window size, occlusions and the left–right check |
| Panorama stitching | Corners → matches → RANSAC homography → warping, with overwrite / average / feather blending |

### Deep learning

Everything trains in the browser with TensorFlow.js, on a 6,000-image MNIST subset or synthetic data.

| Lesson | What you can explore |
|---|---|
| Classification | Train LeNet-5 on MNIST, draw digits to classify, and look inside the network: learned filters, activation maps and occlusion sensitivity |
| Learn a kernel | Gradient descent recovers a hidden 3×3 kernel from input/output images: learning rate, momentum, label noise, ill-conditioning |
| Receptive field | How far a unit sees through stacks of convolutions, pooling, strides and dilation, with path counts (effective receptive field) |
| Saliency | Vanilla gradients, SmoothGrad, Grad-CAM and occlusion for any class, plus the untrained-network sanity check |
| Adversarial | FGSM, targeted and iterative (PGD) attacks on LeNet, and accuracy as a function of ε |
| Overfitting | A perceptron memorising small training sets; dropout, L2, augmentation and data size compared run by run |
| Autoencoder | A 2-D latent space for MNIST: plain vs. variational autoencoder, hover to decode, decoded grid, β |
| U-Net | Segmentation of synthetic shapes where Otsu fails, with and without skip connections |
| Detection | IoU, non-max suppression step by step, precision–recall and AP, anchor assignment |
| Diffusion | The forward noising process on digits (linear vs. cosine schedule), and a denoiser trained to generate 2-D point clouds |
| Vision Transformer | A tiny ViT written with tensor ops: attention maps per head, attention rollout, learned position embeddings |

The lessons that need a trained digit classifier (Saliency, Adversarial) share the LeNet trained in
Classification, or train a quick one themselves.

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
├── App.tsx                  ← header, lesson menu, hash routing, lazy lessons
├── app/                     ← home page, site title/tagline, lesson list and groups, hash router, error boundary
├── image-processing/        ← one folder per image-processing lesson, plus:
│   └── processors/          ←   convolution filters for the Convolution lesson
│       ├── types.ts         ←     Processor interface (the contract)
│       ├── registry.ts      ←     central registry (the UI lists what is registered)
│       ├── init.ts          ←     registers the built-in processors
│       ├── kernels.ts       ←     every shared kernel (Sobel, Prewitt, Laplacian, Gaussian…)
│       └── utils.ts         ←     convolve (replicate padding), output mapping, gray ↔ RGBA
├── deep-learning/           ← one folder per deep-learning lesson, plus:
│   └── shared/              ←   MNIST loading, the LeNet shared between lessons, training helpers,
│                                digit picker, probability bars, colour-mapped rendering
├── hooks/useImageData.ts    ← load a URL or canvas → downscaled RGBA + grayscale pixels
├── components/              ← PixelCanvas, ImageUploader, CodeViewer (+ tokenizer), LineChart
├── lib/                     ← canvas helpers, demo images, colour maps, seeded random
└── styles/base.css          ← design tokens and shared building blocks

<group>/<lesson>/            ← every lesson folder has the same shape:
├── <algorithm>.ts           ←   algorithm code, no DOM (what "Show code" displays)
├── <algorithm>.test.ts      ←   its unit tests
├── <algorithm>.py           ←   the same algorithm in Python / PyTorch (the "Python" tab of "Show code")
├── <Lesson>Section.tsx      ←   UI
└── <lesson>.css             ←   styles
```

Algorithms are plain TypeScript functions over `Float64Array` pixels with no DOM access, so they are
unit-tested in Node. The deep-learning models use TensorFlow.js, whose CPU backend runs the same code
in the tests; TensorFlow.js is only downloaded when a deep-learning lesson is opened. The UI derives everything from state with `useMemo`, and images are drawn
through one `PixelCanvas` component that keeps the aspect ratio.

`scripts/build-mnist-sprite.mjs` packs a balanced MNIST subset (600 train / 100 validation /
100 test images per digit) from the `mnist` package into a 1.2 MB gzip file, so the browser never
downloads the package's 20 MB of JSON.

## Python / PyTorch versions

Every lesson also has its algorithm written with PyTorch tensor operations (`F.conv2d`, `unfold`,
`grid_sample`, `torch.fft`, autograd, `nn.Module`…), shown in the **Python · PyTorch** tab of "Show code".
Each `.py` file is standalone and ends with a self-check on synthetic data:

```bash
pip install -r requirements.txt          # torch (the CPU wheel is enough)
python scripts/check_python.py           # run every self-check
python scripts/check_python.py stereo    # or only the files whose path contains "stereo"
python src/image-processing/stereo/stereo.py
```

CI runs `scripts/check_python.py` in a separate job.

## Adding a new convolution filter

See [`CONTRIBUTING.md`](./CONTRIBUTING.md). In short: create `src/image-processing/processors/your-filter/index.ts`
exporting a `Processor` (kernel, output mode, parameters), register it in `src/image-processing/processors/init.ts`,
and it appears in the Convolution section with controls generated from its parameters.

## Tech stack

React 19 · TypeScript (strict) · Vite · Vitest · oxlint · TensorFlow.js
