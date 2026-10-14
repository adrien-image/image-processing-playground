// Packs a balanced MNIST subset into a compact binary for the Classification section.
//
// Output (generated, git-ignored):
//   public/data/mnist.bin   gzip-compressed Uint8 pixels, one 28×28 image (784 bytes) per row
//   public/data/mnist.json  layout metadata read by src/dl/mnist.ts
//
// Rows are ordered split-major then digit-major, so labels are implicit:
//   train: digit 0 × perDigit.train, digit 1 × perDigit.train, … then val, then test.
// This replaces shipping the `mnist` npm package's 20 MB of JSON to the browser.

import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const require = createRequire(import.meta.url);
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'public', 'data');
const binPath = join(outDir, 'mnist.bin');
const metaPath = join(outDir, 'mnist.json');

const PIXELS = 28 * 28;
const PER_DIGIT = { train: 600, val: 100, test: 100 };
const SPLITS = /** @type {const} */ (['train', 'val', 'test']);

const scriptMtime = statSync(fileURLToPath(import.meta.url)).mtimeMs;
if (existsSync(binPath) && existsSync(metaPath) && statSync(binPath).mtimeMs > scriptMtime) {
  console.log('[mnist] public/data/mnist.bin is up to date');
  process.exit(0);
}

/** @type {number[][]} raw[digit] = flat array of 0–1 floats, 784 per sample */
const raw = [];
for (let d = 0; d < 10; d++) {
  raw.push(require(`mnist/src/digits/${d}.json`).data);
}

const perDigitTotal = PER_DIGIT.train + PER_DIGIT.val + PER_DIGIT.test;
for (let d = 0; d < 10; d++) {
  const available = raw[d].length / PIXELS;
  if (available < perDigitTotal) {
    throw new Error(`[mnist] digit ${d} has only ${available} samples, need ${perDigitTotal}`);
  }
}

const totalRows = perDigitTotal * 10;
const pixels = new Uint8Array(totalRows * PIXELS);
let row = 0;
let offset = 0; // first sample index of the current split within each digit
for (const split of SPLITS) {
  for (let d = 0; d < 10; d++) {
    for (let i = 0; i < PER_DIGIT[split]; i++) {
      const src = (offset + i) * PIXELS;
      for (let p = 0; p < PIXELS; p++) {
        // The package stores byte/255 rounded to 3 decimals; this recovers the original byte.
        pixels[row * PIXELS + p] = Math.round(raw[d][src + p] * 255);
      }
      row++;
    }
  }
  offset += PER_DIGIT[split];
}

mkdirSync(outDir, { recursive: true });
const gz = gzipSync(pixels, { level: 9 });
writeFileSync(binPath, gz);
writeFileSync(metaPath, JSON.stringify({ imageSize: 28, perDigit: PER_DIGIT, splits: SPLITS }, null, 2) + '\n');
console.log(`[mnist] wrote ${totalRows} images (${(gz.length / 1024 / 1024).toFixed(2)} MB gzipped) to public/data/`);
