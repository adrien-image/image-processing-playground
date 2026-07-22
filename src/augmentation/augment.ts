/**
 * Data augmentation engine — pure canvas-based transformations.
 * Each function takes an HTMLImageElement and returns a data URL.
 */

/** Crop a random region then resize back to original size */
export function cropRandom(img: HTMLImageElement, scale: number): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const cropW = Math.round(w * scale);
  const cropH = Math.round(h * scale);
  const x = Math.floor(Math.random() * (w - cropW));
  const y = Math.floor(Math.random() * (h - cropH));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, x, y, cropW, cropH, 0, 0, w, h);
  return c.toDataURL();
}

/** Rotate by a random angle within ±maxAngle degrees, padded output same size */
export function rotate(img: HTMLImageElement, maxAngle: number): string {
  const angle = (Math.random() * 2 - 1) * maxAngle;
  const w = img.naturalWidth, h = img.naturalHeight;
  const rad = angle * Math.PI / 180;
  const cw = Math.ceil(Math.abs(w * Math.cos(rad)) + Math.abs(h * Math.sin(rad)));
  const ch = Math.ceil(Math.abs(w * Math.sin(rad)) + Math.abs(h * Math.cos(rad)));
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);
  ctx.translate(cw / 2, ch / 2);
  ctx.rotate(rad);
  ctx.drawImage(img, -w / 2, -h / 2);

  // Resize back to original size
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const oCtx = out.getContext('2d')!;
  oCtx.drawImage(c, 0, 0, cw, ch, 0, 0, w, h);
  return out.toDataURL();
}

/** Horizontal or vertical flip */
export function flip(img: HTMLImageElement, direction: 'horizontal' | 'vertical'): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  if (direction === 'horizontal') { ctx.translate(w, 0); ctx.scale(-1, 1); }
  else { ctx.translate(0, h); ctx.scale(1, -1); }
  ctx.drawImage(img, 0, 0);
  return c.toDataURL();
}

/** Adjust brightness: value > 1 brightens, < 1 darkens */
export function adjustBrightness(img: HTMLImageElement, factor: number): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.min(255, data[i] * factor);
    data[i + 1] = Math.min(255, data[i + 1] * factor);
    data[i + 2] = Math.min(255, data[i + 2] * factor);
  }
  ctx.putImageData(new ImageData(data, w, h), 0, 0);
  return c.toDataURL();
}

/** Adjust contrast: factor > 1 increases contrast */
export function adjustContrast(img: HTMLImageElement, factor: number): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  const meanRgb = [128, 128, 128];
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.max(0, Math.min(255, meanRgb[0] + (data[i] - meanRgb[0]) * factor));
    data[i + 1] = Math.max(0, Math.min(255, meanRgb[1] + (data[i + 1] - meanRgb[1]) * factor));
    data[i + 2] = Math.max(0, Math.min(255, meanRgb[2] + (data[i + 2] - meanRgb[2]) * factor));
  }
  ctx.putImageData(new ImageData(data, w, h), 0, 0);
  return c.toDataURL();
}

/** Add Gaussian noise */
export function addGaussianNoise(img: HTMLImageElement, std: number): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  for (let i = 0; i < data.length; i += 4) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const noise = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    data[i] = Math.max(0, Math.min(255, data[i] + noise));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
  }
  ctx.putImageData(new ImageData(data, w, h), 0, 0);
  return c.toDataURL();
}

/** Add salt-and-pepper noise */
export function addSaltPepperNoise(img: HTMLImageElement, density: number): string {
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  for (let i = 0; i < data.length; i += 4) {
    const r = Math.random();
    if (r < density / 2) { data[i] = 0; data[i + 1] = 0; data[i + 2] = 0; }
    else if (r < density) { data[i] = 255; data[i + 1] = 255; data[i + 2] = 255; }
  }
  ctx.putImageData(new ImageData(data, w, h), 0, 0);
  return c.toDataURL();
}

/** Shear transform */
export function shear(img: HTMLImageElement, maxShear: number): string {
  const shearX = (Math.random() * 2 - 1) * maxShear;
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.transform(1, 0, shearX, 1, 0, 0);
  ctx.drawImage(img, 0, 0);
  return c.toDataURL();
}

export interface AugmentConfig {
  type: string;
  label: string;
  apply: (img: HTMLImageElement) => string;
}

/** Generate a config array with random parameters */
export function generateAugmentConfigs(opts: {
  rotate: boolean; flip: boolean; crop: boolean; brightness: boolean;
  contrast: boolean; gaussianNoise: boolean; saltPepper: boolean; shear: boolean;
}): AugmentConfig[] {
  const configs: AugmentConfig[] = [];

  if (opts.rotate) {
    const angle = Math.round(5 + Math.random() * 25);
    configs.push({
      type: 'rotate', label: `Rotate ±${angle}°`,
      apply: (img) => rotate(img, angle),
    });
  }
  if (opts.flip) {
    const dir = Math.random() > 0.5 ? 'horizontal' : 'vertical';
    configs.push({
      type: 'flip', label: `Flip ${dir}`,
      apply: (img) => flip(img, dir),
    });
  }
  if (opts.crop) {
    const s = Math.round((0.7 + Math.random() * 0.2) * 100) / 100;
    configs.push({
      type: 'crop', label: `Crop ${(s * 100).toFixed(0)}%`,
      apply: (img) => cropRandom(img, s),
    });
  }
  if (opts.brightness) {
    const b = Math.round((0.5 + Math.random() * 1.0) * 100) / 100;
    configs.push({
      type: 'brightness', label: `Brightness ×${b.toFixed(1)}`,
      apply: (img) => adjustBrightness(img, b),
    });
  }
  if (opts.contrast) {
    const c = Math.round((0.5 + Math.random() * 1.5) * 100) / 100;
    configs.push({
      type: 'contrast', label: `Contrast ×${c.toFixed(1)}`,
      apply: (img) => adjustContrast(img, c),
    });
  }
  if (opts.gaussianNoise) {
    const std = Math.round(5 + Math.random() * 30);
    configs.push({
      type: 'gaussian', label: `Gaussian σ=${std}`,
      apply: (img) => addGaussianNoise(img, std),
    });
  }
  if (opts.saltPepper) {
    const d = Math.round((0.02 + Math.random() * 0.15) * 100) / 100;
    configs.push({
      type: 'saltpepper', label: `S&P ${(d * 100).toFixed(0)}%`,
      apply: (img) => addSaltPepperNoise(img, d),
    });
  }
  if (opts.shear) {
    const s = Math.round((0.05 + Math.random() * 0.25) * 100) / 100;
    configs.push({
      type: 'shear', label: `Shear ×${s.toFixed(2)}`,
      apply: (img) => shear(img, s),
    });
  }

  return configs;
}
