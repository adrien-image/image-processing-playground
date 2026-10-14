import { useEffect, useMemo, useState } from 'react';
import { convolve, makeOutputMapper, type OutputMode } from '../processors/utils';
import { grayToImageData } from '../lib/image';
import type { LoadedImage } from '../hooks/useImageData';
import { revealPartial, snakeOrder, stepMath } from './playback';

/** Playback speed bounds, in pixels per second. */
export const MIN_SPEED = 1;
export const MAX_SPEED = 400;
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
 */
export function useConvolutionPlayback(
  image: LoadedImage | null,
  kernel: number[][] | null,
  mode: OutputMode | null,
  resetKey: string,
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

  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(20); // pixels per second

  // Rewind when the image or processor changes (adjusting state during render,
  // per https://react.dev/learn/you-might-not-need-an-effect).
  const [lastResetKey, setLastResetKey] = useState(resetKey);
  if (lastResetKey !== resetKey) {
    setLastResetKey(resetKey);
    setStepIndex(0);
    setPlaying(false);
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
      partial: grayToImageData(revealPartial(output, order, index + 1, width), width, height),
      math: stepMath(gray, width, height, k, pos, map),
    };
  }, [prepared, index]);

  const goTo = (i: number) => {
    setStepIndex(Math.max(0, Math.min(i, total - 1)));
  };

  const togglePlay = () => {
    if (!playing && index >= total - 1) setStepIndex(0);
    setPlaying(p => !p);
  };

  return { ...view, index, total, playing, speed, setSpeed, goTo, togglePlay, stop: () => setPlaying(false) };
}
