import { useMemo, useState } from 'react';
import convolveCode from '../processors/utils.ts?raw';
import pythonCode from './convolution.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { checkerboard, circle, corner, scene, verticalEdge } from '../../lib/demoImages';
import { getProcessor, resolveConfig, type ProcessorConfig } from '../processors';
import { ImageSourceSelector, type Resolution } from './ImageSourceSelector';
import { PixelPicker } from './PixelPicker';
import { snakeIndex, type PixelPos } from './playback';
import type { DemoImage } from './demoImageNames';
import { ProcessorSelector } from './ProcessorSelector';
import { KernelMatrixDisplay } from './KernelMatrixDisplay';
import { PlaybackControls } from './PlaybackControls';
import { MathPanel } from './MathPanel';
import { useConvolutionPlayback } from './useConvolutionPlayback';
import './convolution.css';

const DEMOS: Record<DemoImage, (size: number) => HTMLCanvasElement> = {
  scene: (size) => scene(size),
  checkerboard: (size) => checkerboard(size, size / 4),
  circle: (size) => circle(size),
  edges: (size) => verticalEdge(size),
  corner: (size) => corner(size),
};

/**
 * Small images keep every pixel and every number readable while stepping through;
 * large ones show what the filter actually does to a picture.
 */
const RESOLUTIONS: Record<Resolution, { demo: number; maxUpload: number; startAtEnd: boolean; speed: number }> = {
  small: { demo: 32, maxUpload: 64, startAtEnd: false, speed: 20 },
  large: { demo: 256, maxUpload: 256, startAtEnd: true, speed: 2000 },
};

const DISPLAY_SIZE = 480;

type IpTab = 'processing' | 'computation';

export function ConvolutionSection() {
  const [tab, setTab] = useState<IpTab>('processing');
  const [showCode, setShowCode] = useState(false);
  const [resolution, setResolution] = useState<Resolution>('large');
  const res = RESOLUTIONS[resolution];
  const [demo, setDemo] = useState<DemoImage | 'upload'>('scene');
  const [demoSource, setDemoSource] = useState<ImageSource>(() => DEMOS.scene(res.demo));
  const [upload, setUpload] = useState<string | null>(null);
  const source = demo === 'upload' && upload ? upload : demoSource;
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  // Remember parameter values per processor, so switching back keeps them.
  const [configs, setConfigs] = useState<Record<string, ProcessorConfig>>({});

  const { image, error: loadError, loading } = useImageData(source, res.maxUpload);
  const processor = activeId ? getProcessor(activeId) ?? null : null;
  const config = useMemo(
    () => (processor ? resolveConfig(processor, configs[processor.id] ?? {}) : {}),
    [processor, configs],
  );
  const kernel = useMemo(() => (processor ? processor.kernel(config) : null), [processor, config]);

  const playback = useConvolutionPlayback(
    image,
    kernel,
    processor?.output ?? null,
    `${demo}|${resolution}|${upload ?? ''}|${activeId}`,
    { startAtEnd: res.startAtEnd, defaultSpeed: res.speed },
  );

  const selectDemo = (name: DemoImage) => {
    setDemo(name);
    setDemoSource(DEMOS[name](res.demo));
    setUploadError(null);
  };
  const selectUpload = (src: string) => {
    setDemo('upload');
    setUpload(src);
    setUploadError(null);
  };
  const selectResolution = (next: Resolution) => {
    setResolution(next);
    if (demo !== 'upload') setDemoSource(DEMOS[demo](RESOLUTIONS[next].demo));
  };
  const pick = (p: PixelPos) => {
    playback.stop();
    playback.goTo(snakeIndex(p, width), { inspect: true });
  };

  const error = uploadError ?? loadError;
  const width = image?.width ?? 0;
  const height = image?.height ?? 0;
  const pos = processor ? playback.position : undefined;
  const half = kernel ? (kernel.length - 1) >> 1 : 1;
  const pct = (v: number, of: number) => `${(v / of) * 100}%`;

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <ImageSourceSelector
          active={demo}
          resolution={resolution}
          onSelect={selectDemo}
          onUpload={selectUpload}
          onResolutionChange={selectResolution}
          onError={setUploadError}
        />
        <ProcessorSelector
          activeProcessor={activeId}
          config={config}
          onSelect={setActiveId}
          onConfigChange={(c) => activeId && setConfigs(prev => ({ ...prev, [activeId]: c }))}
          disabled={!image}
        />
        {processor && kernel && <KernelMatrixDisplay kernel={kernel} label={processor.name} />}
        {error && <div className="error-banner" role="alert">⚠ {error}</div>}
        {loading && <div className="loading" role="status">⏳ Loading…</div>}
      </aside>

      <div className="main-content">
        <div className="tab-bar">
          <div className="tab-list" role="tablist" aria-label="Convolution views">
            <button type="button" role="tab" id="conv-tab-processing" aria-controls="conv-panel"
              aria-selected={tab === 'processing'} className={`tab-btn${tab === 'processing' ? ' tab-active' : ''}`}
              onClick={() => setTab('processing')}>Processing</button>
            <button type="button" role="tab" id="conv-tab-computation" aria-controls="conv-panel"
              aria-selected={tab === 'computation'} className={`tab-btn${tab === 'computation' ? ' tab-active' : ''}`}
              onClick={() => setTab('computation')}>Computation</button>
          </div>
          <ShowCodeButton code={convolveCode} title="processors/utils.ts" python={{ code: pythonCode, title: 'convolution/convolution.py' }} open={showCode} onOpenChange={setShowCode} />
        </div>

        <div id="conv-panel" role="tabpanel" aria-labelledby={`conv-tab-${tab}`} className="conv-panel">
          {tab === 'processing' && (
            <>
              <div className="image-display">
                <div className="image-panel">
                  <h3>Original</h3>
                  <PixelCanvas image={image?.imageData ?? null} size={DISPLAY_SIZE} label="Original image">
                    {pos && (
                      <>
                        <div
                          className="conv-window"
                          style={{
                            left: pct(pos.col - half, width), top: pct(pos.row - half, height),
                            width: pct(2 * half + 1, width), height: pct(2 * half + 1, height),
                          }}
                        />
                        <div
                          className="conv-window-centre"
                          style={{ left: pct(pos.col, width), top: pct(pos.row, height), width: pct(1, width), height: pct(1, height) }}
                        />
                        <div className="position-badge">({pos.col}, {pos.row})</div>
                        <PixelPicker
                          width={width} height={height} position={pos}
                          index={playback.index} total={playback.total}
                          label="Original image: pick a pixel to inspect" onPick={pick}
                        />
                      </>
                    )}
                  </PixelCanvas>
                  <span className="dimensions">{width}×{height}</span>
                </div>

                <div className="image-panel">
                  <h3>Processed</h3>
                  <PixelCanvas
                    image={processor ? playback.partial ?? null : null}
                    size={DISPLAY_SIZE}
                    placeholderAspect={width && height ? width / height : 1}
                    label={processor ? `${processor.name} output` : 'No processor selected'}
                    placeholder="Select a processor"
                  >
                    {pos && (
                      <div
                        className="conv-window-centre conv-window-output"
                        style={{ left: pct(pos.col, width), top: pct(pos.row, height), width: pct(1, width), height: pct(1, height) }}
                      />
                    )}
                    {pos && (
                      <PixelPicker
                        width={width} height={height} position={pos}
                        index={playback.index} total={playback.total}
                        label="Processed image: pick a pixel to inspect" onPick={pick}
                      />
                    )}
                  </PixelCanvas>
                  <span className="dimensions">{width}×{height}</span>
                </div>
              </div>

              {processor && (
                <p className="pick-hint">
                  Click or drag on either image to inspect any pixel, or press ▶ to watch the kernel sweep the image.
                  {resolution === 'large' && ' Switch to Small (32 px) to follow every step of the arithmetic.'}
                </p>
              )}

              {processor && playback.total > 0 && (
                <PlaybackControls
                  index={playback.index}
                  total={playback.total}
                  playing={playback.playing}
                  speed={playback.speed}
                  position={playback.position}
                  onGoTo={(i) => { playback.stop(); playback.goTo(i); }}
                  onTogglePlay={playback.togglePlay}
                  onSpeedChange={playback.setSpeed}
                />
              )}
              {processor && playback.math && (
                <div className="tab-content-computation">
                  <MathPanel data={playback.math} outputMode={processor.output} />
                </div>
              )}
            </>
          )}

          {tab === 'computation' && (
            <div className="tab-content-computation">
              {processor && playback.math ? (
                <>
                  <p className="computation-position mono">
                    Pixel ({playback.position?.col}, {playback.position?.row}) · step {playback.index + 1} / {playback.total}
                  </p>
                  <MathPanel data={playback.math} outputMode={processor.output} />
                </>
              ) : (
                <div className="placeholder computation-placeholder">
                  Select a processor to see how each output pixel is computed.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
