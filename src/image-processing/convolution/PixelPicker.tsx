import type { KeyboardEvent, PointerEvent } from 'react';
import type { PixelPos } from './playback';

interface PixelPickerProps {
  width: number;
  height: number;
  position: PixelPos | undefined;
  /** Step index of `position`, for assistive technology. */
  index: number;
  total: number;
  label: string;
  onPick: (pos: PixelPos) => void;
}

/**
 * Transparent layer over an image: click or drag to jump the convolution window to
 * any pixel, or focus it and move with the arrow keys (Shift moves 8 pixels).
 */
export function PixelPicker({ width, height, position, index, total, label, onPick }: PixelPickerProps) {
  const pickAt = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * width);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * height);
    onPick({ col: clamp(col, 0, width - 1), row: clamp(row, 0, height - 1) });
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    pickAt(e);
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) pickAt(e);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 8 : 1;
    const delta: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step],
    };
    const d = delta[e.key];
    if (!d) return;
    e.preventDefault();
    const { col, row } = position ?? { col: 0, row: 0 };
    onPick({ col: clamp(col + d[0], 0, width - 1), row: clamp(row + d[1], 0, height - 1) });
  };

  return (
    <div
      className="pixel-picker"
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={1}
      aria-valuemax={total}
      aria-valuenow={index + 1}
      aria-valuetext={position ? `Pixel (${position.col}, ${position.row})` : undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onKeyDown={onKeyDown}
    />
  );
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
