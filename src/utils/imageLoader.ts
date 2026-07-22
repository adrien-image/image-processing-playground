/**
 * Load an image from a URL / path into an HTMLImageElement.
 */
export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

/**
 * Draw an image onto a canvas and extract its ImageData.
 */
export function getImageData(
  img: HTMLImageElement,
  maxSize = 512,
): { imageData: ImageData; width: number; height: number } {
  let { width, height } = img;

  // Downscale if larger than maxSize (maintain aspect ratio)
  if (width > maxSize || height > maxSize) {
    const ratio = Math.min(maxSize / width, maxSize / height);
    width = Math.round(width * ratio);
    height = Math.round(height * ratio);
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  return { imageData, width, height };
}

export function generateEdgesDataURL(size = 32): string {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Dark background
  ctx.fillStyle = '#222';
  ctx.fillRect(0, 0, size, size);

  // White vertical stripe on left half
  ctx.fillStyle = '#fff';
  ctx.fillRect(4, 0, size / 2 - 4, size);

  return canvas.toDataURL();
}

/**
 * Generate a corner pattern — white top-left triangle on dark bg.
 */
export function generateCornerDataURL(size = 32): string {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#222';
  ctx.fillRect(0, 0, size, size);

  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(size, 0);
  ctx.lineTo(0, size);
  ctx.closePath();
  ctx.fill();

  return canvas.toDataURL();
}

/**
 * Generate a simple checkerboard pattern as a data URL.
 * Used as a fallback/utility demo image.
 */
export function generateCheckerboardDataURL(size = 32, tileSize = 8): string {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const isWhite = (Math.floor(x / tileSize) + Math.floor(y / tileSize)) % 2 === 0;
      ctx.fillStyle = isWhite ? '#fff' : '#333';
      ctx.fillRect(x, y, 1, 1);
    }
  }

  return canvas.toDataURL();
}

/**
 * Generate a simple circle on a solid background.
 */
export function generateCircleDataURL(size = 32): string {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Background
  ctx.fillStyle = '#333';
  ctx.fillRect(0, 0, size, size);

  // Circle
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size * 0.35, 0, Math.PI * 2);
  ctx.fill();

  return canvas.toDataURL();
}
