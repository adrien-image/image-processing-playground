import { createCanvas } from './canvas';

/** Draw a synthetic demo image into a fresh canvas. */
export function makeCanvas(width: number, height: number, draw: (ctx: CanvasRenderingContext2D) => void): HTMLCanvasElement {
  const { canvas, ctx } = createCanvas(width, height);
  draw(ctx);
  return canvas;
}

/** Checkerboard of `tileSize` px squares (white / #333). */
export function checkerboard(size = 32, tileSize = 8): HTMLCanvasElement {
  return makeCanvas(size, size, ctx => {
    for (let ty = 0; ty < size; ty += tileSize) {
      for (let tx = 0; tx < size; tx += tileSize) {
        ctx.fillStyle = ((tx / tileSize + ty / tileSize) % 2 === 0) ? '#fff' : '#333';
        ctx.fillRect(tx, ty, tileSize, tileSize);
      }
    }
  });
}

/** White disc on a dark background. */
export function circle(size = 32): HTMLCanvasElement {
  return makeCanvas(size, size, ctx => {
    ctx.fillStyle = '#333';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size * 0.35, 0, Math.PI * 2);
    ctx.fill();
  });
}

/** White vertical stripe on a dark background (two vertical edges). */
export function verticalEdge(size = 32): HTMLCanvasElement {
  return makeCanvas(size, size, ctx => {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#fff';
    ctx.fillRect(size / 8, 0, size * 3 / 8, size);
  });
}

/** White top-left triangle on a dark background (a corner and a diagonal edge). */
export function corner(size = 32): HTMLCanvasElement {
  return makeCanvas(size, size, ctx => {
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(size, 0);
    ctx.lineTo(0, size);
    ctx.closePath();
    ctx.fill();
  });
}

/**
 * A small still life: a soft gradient, smooth shapes, a hard-edged box, fine stripes
 * and text. Blur, sharpen and edge filters each change a different part of it, so
 * it shows their effect far better than a single edge does at higher resolutions.
 */
export function scene(size = 256): HTMLCanvasElement {
  return makeCanvas(size, size, ctx => {
    const u = size / 32; // 1 unit = 1/32 of the image
    const sky = ctx.createLinearGradient(0, 0, 0, size);
    sky.addColorStop(0, '#2a3348');
    sky.addColorStop(1, '#8a93a8');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, size, size);

    // Sun: a bright disc with a soft halo.
    const halo = ctx.createRadialGradient(9 * u, 9 * u, 3 * u, 9 * u, 9 * u, 8 * u);
    halo.addColorStop(0, 'rgba(255,255,255,0.55)');
    halo.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#f4f4f4';
    ctx.beginPath();
    ctx.arc(9 * u, 9 * u, 4 * u, 0, Math.PI * 2);
    ctx.fill();

    // Hard-edged dark box with a bright window.
    ctx.fillStyle = '#16181f';
    ctx.fillRect(18 * u, 12 * u, 11 * u, 14 * u);
    ctx.fillStyle = '#e6e6e6';
    ctx.fillRect(20 * u, 14 * u, 3 * u, 3 * u);
    ctx.fillRect(25 * u, 14 * u, 2 * u, 7 * u);

    // Mountain: a diagonal edge in two directions.
    ctx.fillStyle = '#3c4152';
    ctx.beginPath();
    ctx.moveTo(0, 28 * u);
    ctx.lineTo(8 * u, 17 * u);
    ctx.lineTo(16 * u, 28 * u);
    ctx.closePath();
    ctx.fill();

    // Fine stripes: high-frequency detail that blur removes and sharpen boosts.
    ctx.fillStyle = '#dcdcdc';
    const stripe = Math.max(1, Math.round(u / 2));
    for (let x = 2 * u; x < 14 * u; x += 2 * stripe) ctx.fillRect(Math.round(x), 29 * u, stripe, 2.5 * u);

    // Text: thin strokes and corners.
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${Math.round(4 * u)}px sans-serif`;
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('Aa', 18 * u, 31 * u);
  });
}
