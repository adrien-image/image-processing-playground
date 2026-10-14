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
    ctx.fillRect(4, 0, size / 2 - 4, size);
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
