/**
 * Loads the compact MNIST subset produced by `scripts/build-mnist-sprite.mjs`
 * (public/data/mnist.bin + mnist.json): 600 train, 100 validation and 100 test
 * images per digit, stored as gzip-compressed bytes.
 */

export interface MnistSplit {
  /** n × 784 pixel values in [0, 1] */
  images: Float32Array;
  /** n digit labels (0–9) */
  labels: Uint8Array;
  count: number;
}

export interface MnistData {
  train: MnistSplit;
  val: MnistSplit;
  test: MnistSplit;
}

interface MnistMeta {
  imageSize: number;
  perDigit: Record<'train' | 'val' | 'test', number>;
  splits: ('train' | 'val' | 'test')[];
}

const PIXELS = 28 * 28;

/** Undo gzip unless the server already did (some send `Content-Encoding: gzip`). */
async function gunzip(buffer: ArrayBuffer): Promise<Uint8Array> {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Split the row-ordered bytes into labelled sets (rows are split-major, then digit-major). */
export function decodeMnist(bytes: Uint8Array, meta: MnistMeta): MnistData {
  const out = {} as MnistData;
  let row = 0;
  for (const split of meta.splits) {
    const per = meta.perDigit[split];
    const count = per * 10;
    const images = new Float32Array(count * PIXELS);
    const labels = new Uint8Array(count);
    for (let i = 0; i < count; i++, row++) {
      labels[i] = Math.floor(i / per);
      for (let p = 0; p < PIXELS; p++) images[i * PIXELS + p] = bytes[row * PIXELS + p] / 255;
    }
    out[split] = { images, labels, count };
  }
  if (row * PIXELS !== bytes.length) throw new Error('MNIST data does not match its metadata');
  return out;
}

export async function loadMnist(baseUrl = import.meta.env.BASE_URL): Promise<MnistData> {
  const [metaRes, binRes] = await Promise.all([fetch(`${baseUrl}data/mnist.json`), fetch(`${baseUrl}data/mnist.bin`)]);
  if (!metaRes.ok || !binRes.ok) {
    throw new Error('MNIST data not found — run `npm run prepare-data` (it runs automatically before dev/build).');
  }
  const meta = (await metaRes.json()) as MnistMeta;
  return decodeMnist(await gunzip(await binRes.arrayBuffer()), meta);
}
