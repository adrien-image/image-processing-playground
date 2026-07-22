# 🧠 Image Processing Playground

An interactive web app that teaches how image processing works — starting with **convolution** and **Sobel edge detection**.

## Features

- **Step-by-step convolution animation** — watch the kernel slide pixel-by-pixel
- **Sobel Edge Detection** — with adjustable threshold and normalisation
- **Built-in demo images** (checkerboard, circle) plus user upload
- **Extensible processor architecture** — add new filters in minutes

## Quick Start

```bash
npm install
npm run dev
```

## Architecture

```
src/
├── processors/          ← All image processing logic
│   ├── types.ts         ← Processor interface (the contract)
│   ├── registry.ts      ← Central registry (UI auto-discovers)
│   ├── init.ts          ← Registration bootstrap
│   ├── utils.ts         ← Convolution math helpers
│   └── sobel/           ← Sobel edge detection
├── components/
│   ├── ImageSourceSelector.tsx
│   ├── ProcessorSelector.tsx
│   ├── ImageDisplay.tsx
│   └── ConvolutionAnimation.tsx
└── App.tsx
```

## Adding a New Processor

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for full instructions.

In short: create a file in `src/processors/your-filter/` that exports a `Processor` object, register it in `init.ts`, and the UI discovers it automatically.

## Tech Stack

React + TypeScript + Vite
# image-processing-playground
