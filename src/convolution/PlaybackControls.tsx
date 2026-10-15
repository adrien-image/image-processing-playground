import { useId } from 'react';
import { MAX_SPEED, MIN_SPEED } from './useConvolutionPlayback';
import type { PixelPos } from './playback';

interface PlaybackControlsProps {
  index: number;
  total: number;
  playing: boolean;
  speed: number;
  position: PixelPos | undefined;
  onGoTo: (index: number) => void;
  onTogglePlay: () => void;
  onSpeedChange: (pixelsPerSecond: number) => void;
}

/** Transport controls for the pixel-by-pixel convolution animation. */
export function PlaybackControls({ index, total, playing, speed, position, onGoTo, onTogglePlay, onSpeedChange }: PlaybackControlsProps) {
  const speedId = useId();
  const atStart = index === 0;
  const atEnd = index >= total - 1;

  return (
    <div className="progressive-processing">
      <div className="proc-controls">
        <div className="proc-controls-left">
          <button type="button" className="btn btn-sm" onClick={() => onGoTo(0)} disabled={atStart} aria-label="First pixel">⏮</button>
          <button type="button" className="btn btn-sm" onClick={() => onGoTo(index - 1)} disabled={atStart} aria-label="Previous pixel">◀</button>
          <button type="button" className="btn btn-sm" onClick={onTogglePlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? '⏸' : '▶'}</button>
          <button type="button" className="btn btn-sm" onClick={() => onGoTo(index + 1)} disabled={atEnd} aria-label="Next pixel">▶</button>
          <button type="button" className="btn btn-sm" onClick={() => onGoTo(total - 1)} disabled={atEnd} aria-label="Last pixel">⏭</button>
        </div>

        <div className="proc-info" aria-live="off">
          <span className="proc-pixel">{position ? `Pixel (${position.col}, ${position.row})` : '—'}</span>
          <span className="proc-progress">{total > 0 ? `${index + 1} / ${total}` : '—'}</span>
        </div>

        <div className="proc-controls-right">
          <div className="speed-control">
            <label htmlFor={speedId}>Speed</label>
            <input
              id={speedId}
              type="range"
              min={Math.log(MIN_SPEED)}
              max={Math.log(MAX_SPEED)}
              step={0.01}
              value={Math.log(speed)}
              onChange={(e) => onSpeedChange(Math.round(Math.exp(Number(e.target.value))))}
              aria-valuetext={`${speed.toLocaleString('en')} pixels per second`}
            />
            <span className="speed-value">{speed.toLocaleString('en')} px/s</span>
          </div>
        </div>
      </div>

      <div className="proc-progress-bar">
        <input
          type="range"
          min={0}
          max={Math.max(total - 1, 0)}
          value={index}
          onChange={(e) => onGoTo(Number(e.target.value))}
          aria-label="Animation position"
        />
      </div>
    </div>
  );
}
