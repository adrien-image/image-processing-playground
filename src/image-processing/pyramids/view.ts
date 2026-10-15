/** Browser-side display helpers for the Pyramids section (the algorithms stay DOM-free). */
import { toImageData } from '../../lib/canvas';
import { makeCanvas } from '../../lib/demoImages';
import { seaScene, sunsetScene, zonePlate } from './demos';
import { mergeRGB, type Plane } from './plane';
import { gaussianPyramid, laplacianFromGaussian } from './pyramid';

/** Planes (1 = gray, 3 = RGB) → ImageData, each value shown as `offset + gain · v`. */
export function planesToImageData(planes: readonly Plane[], offset = 0, gain = 1): ImageData {
  return toImageData(mergeRGB(planes, offset, gain), planes[0].width, planes[0].height);
}

/** Pyramids are computed per channel ([channel][level]); displays need [level][channel]. */
export function byLevel(perChannel: readonly (readonly Plane[])[]): Plane[][] {
  return perChannel[0].map((_, level) => perChannel.map(channel => channel[level]));
}

/** Per channel: a Gaussian pyramid and the Laplacian pyramid built from it ([channel][level]). */
export function pyramidsPerChannel(planes: readonly Plane[], levels: number): { gaussian: Plane[][]; laplacian: Plane[][] } {
  const gaussian = planes.map(p => gaussianPyramid(p, levels));
  return { gaussian, laplacian: gaussian.map(laplacianFromGaussian) };
}

function rgbaCanvas(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): HTMLCanvasElement {
  return makeCanvas(width, height, ctx => ctx.putImageData(toImageData(rgba, width, height), 0, 0));
}

export const DEMO_SIZE = 256;

export const zonePlateCanvas = () => rgbaCanvas(mergeRGB([zonePlate(DEMO_SIZE)]), DEMO_SIZE, DEMO_SIZE);
export const sunsetCanvas = () => rgbaCanvas(sunsetScene(DEMO_SIZE, DEMO_SIZE), DEMO_SIZE, DEMO_SIZE);
export const seaCanvas = () => rgbaCanvas(seaScene(DEMO_SIZE, DEMO_SIZE), DEMO_SIZE, DEMO_SIZE);
