import { useEffect, useMemo, useState } from 'react';
import { convolve, makeOutputMapper, type OutputMode } from '../processors/utils';
import { grayToImageData } from '../lib/image';
import type { LoadedImage } from '../hooks/useImageData';
import { revealPartial, snakeOrder, stepMath } from './playback';

/** Playback speed bounds, in pixels per second. */
export const MIN_SPEED = 1;
export const MAX_SPEED = 20_000;
const FRAME_MS = 16;

/**
 * State machine for the step-by-step convolution animation.
 *
 * Everything shown (partial output, highlighted pixel, math) is derived from
 * `stepIndex`, so scrubbing, stepping and playing can never disagree. Lives in the
 * section, so switching between the Processing and Computation tabs keeps the position.
 *
 * `resetKey` identifies the image + processor: when it changes playback rewinds; a
 * kernel change from a parameter slider keeps the current position.
 *
 * Options:
 * - `startAtEnd`: after a reset, show the finished output instead of a black image
 *   (for large images, where watching every pixel would take minutes).
 * - `defaultSpeed`: the speed to switch to whenever this value changes.
 */
export function useConvolutionPlayback(
  image: LoadedImage | null,
  kernel: number[][] | null,
  mode: OutputMode | null,
  resetKey: string,
  { startAtEnd = false, defaultSpeed = 20 }: { startAtEnd?: boolean; defaultSpeed?: number } = {},
) {
  const prepared = useMemo(() => {
    if (!image || !kernel || !mode) return null;
    const { gray, width, height } = image;
    const raw = convolve(gray, width, height, kernel);
    const map = makeOutputMapper(raw, mode);
    const output = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) output[i] = map(raw[i]);
    return { gray, width, height, kernel, map, output, order: snakeOrder(width, height) };
  }, [image, kernel, mode]);

  const firstStep = startAtEnd ? Number.MAX_SAFE_INTEGER : 0; // clamped to the last pixel below
  const [stepIndex, setStepIndex] = useState(firstStep);
  const [playing, setPlaying] = useState(false);
  // True after picking a pixel directly: show the whole output rather than the sweep so far.
  const [revealAll, setRevealAll] = useState(false);
  const [speed, setSpeed] = useState(defaultSpeed); // pixels per second

  // Rewind when the image or processor changes (adjusting state during render,
  // per https://react.dev/learn/you-might-not-need-an-effect).
  const [lastResetKey, setLastResetKey] = useState(resetKey);
  if (lastResetKey !== resetKey) {
    setLastResetKey(resetKey);
    setStepIndex(firstStep);
    setPlaying(false);
    setRevealAll(false);
  }
  const [lastDefaultSpeed, setLastDefaultSpeed] = useState(defaultSpeed);
  if (lastDefaultSpeed !== defaultSpeed) {
    setLastDefaultSpeed(defaultSpeed);
    setSpeed(defaultSpeed);
  }

  const total = prepared?.order.length ?? 0;
  const index = Math.min(stepIndex, Math.max(total - 1, 0));

  useEffect(() => {
    if (!playing || total === 0) return;
    const delay = Math.max(FRAME_MS, 1000 / speed);
    const stepsPerTick = Math.max(1, Math.round((speed * delay) / 1000));
    const timer = window.setTimeout(() => {
      const next = Math.min(index + stepsPerTick, total - 1);
      setStepIndex(next);
      if (next >= total - 1) setPlaying(false);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [playing, index, total, speed]);

  const view = useMemo(() => {
    if (!prepared) return null;
    const { gray, width, height, kernel: k, map, output, order } = prepared;
    const pos = order[index];
    return {
      position: pos,
      partial: grayToImageData(revealAll ? output : revealPartial(output, order, index + 1, width), width, height),
      math: stepMath(gray, width, height, k, pos, map),
    };
  }, [prepared, index, revealAll]);

  /** Jump to step `i`. With `inspect`, keep the full output visible (picking a pixel). */
  const goTo = (i: number, { inspect = false } = {}) => {
    setStepIndex(Math.max(0, Math.min(i, total - 1)));
    setRevealAll(inspect);
  };

  const togglePlay = () => {
    setRevealAll(false);
    if (!playing && index >= total - 1) setStepIndex(0);
    setPlaying(p => !p);
  };

  return { ...view, index, total, playing, speed, setSpeed, goTo, togglePlay, stop: () => setPlaying(false) };
}
