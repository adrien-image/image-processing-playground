import { useMemo, useState, type ChangeEvent } from 'react';
import colorCode from './color.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { toImageData } from '../../lib/canvas';
import './color.css';
import { adjustColors, extractChannel, NO_ADJUSTMENT, type Channel, type ColorAdjustments } from './color';

function demoImage(): HTMLCanvasElement {
  return makeCanvas(200, 200, ctx => {
    ctx.fillStyle = '#ff3333'; ctx.fillRect(0, 0, 100, 100);
    ctx.fillStyle = '#33ff33'; ctx.fillRect(100, 0, 100, 100);
    ctx.fillStyle = '#3333ff'; ctx.fillRect(0, 100, 100, 100);
    ctx.fillStyle = '#ffff33'; ctx.fillRect(100, 100, 100, 100);
    ctx.fillStyle = '#ff88ff'; ctx.beginPath(); ctx.arc(100, 100, 40, 0, Math.PI * 2); ctx.fill();
  });
}

type View = 'rgb' | Channel;

const VIEWS: { key: View; button: string; heading: string }[] = [
  { key: 'rgb', button: 'RGB', heading: 'Processed' },
  { key: 'r', button: 'R', heading: 'Red channel' },
  { key: 'g', button: 'G', heading: 'Green channel' },
  { key: 'b', button: 'B', heading: 'Blue channel' },
  { key: 'h', button: 'Hue', heading: 'Hue' },
  { key: 's', button: 'Sat', heading: 'Saturation' },
  { key: 'v', button: 'Val', heading: 'Value' },
];

const PRESETS: { label: string; adj: Partial<ColorAdjustments> }[] = [
  { label: '↺ Reset', adj: {} },
  { label: '🖤 Desaturate', adj: { satScale: 0 } },
  { label: '🔄 Invert Hue', adj: { hueShift: 180 } },
  { label: '☀ Overexpose', adj: { valScale: 1.8 } },
  { label: '🌙 Darken', adj: { gamma: 0.45 } },
];

const DISPLAY = 260;

export function ColorSection() {
  const [view, setView] = useState<View>('rgb');
  const [adj, setAdj] = useState<ColorAdjustments>(NO_ADJUSTMENT);
  const [showCode, setShowCode] = useState(false);
  const [isDemo, setIsDemo] = useState(true);
  const [source, setSource] = useState<ImageSource>(demoImage);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { image, error: loadError } = useImageData(source, 512);

  const processed = useMemo(() => {
    if (!image) return null;
    const { imageData, width, height } = image;
    const rgba = view === 'rgb' ? adjustColors(imageData.data, adj) : extractChannel(imageData.data, view);
    return toImageData(rgba, width, height);
  }, [image, view, adj]);

  const set = (key: keyof ColorAdjustments) => (e: ChangeEvent<HTMLInputElement>) =>
    setAdj(prev => ({ ...prev, [key]: Number(e.target.value) }));

  const error = uploadError ?? loadError;
  const heading = VIEWS.find(v => v.key === view)!.heading;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Channel:</span>
        {VIEWS.map(v => (
          <button key={v.key} type="button" aria-pressed={view === v.key}
            className={`btn btn-sm${view === v.key ? ' btn-primary' : ''}`} onClick={() => setView(v.key)}>
            {v.button}
          </button>
        ))}
        <ImageUploader
          onImage={(src) => { setSource(src); setIsDemo(false); setUploadError(null); }}
          onError={setUploadError}
        />
        {!isDemo && (
          <button type="button" className="btn btn-sm" onClick={() => { setSource(demoImage()); setIsDemo(true); }}>
            Demo image
          </button>
        )}
        <span className="ml-auto">
          <ShowCodeButton code={colorCode} title="color/color.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      {view === 'rgb' && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="color-hue">Hue shift: {adj.hueShift}°</label>
            <input id="color-hue" type="range" min={0} max={359} value={adj.hueShift} onChange={set('hueShift')} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="color-sat">Saturation: ×{adj.satScale.toFixed(2)}</label>
            <input id="color-sat" type="range" min={0} max={2} step={0.05} value={adj.satScale} onChange={set('satScale')} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="color-val">Brightness (V): ×{adj.valScale.toFixed(2)}</label>
            <input id="color-val" type="range" min={0} max={2} step={0.05} value={adj.valScale} onChange={set('valScale')} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <label className="demo-slider-label" htmlFor="color-gamma">Gamma: {adj.gamma.toFixed(2)}</label>
            <input id="color-gamma" type="range" min={0.1} max={3} step={0.05} value={adj.gamma} onChange={set('gamma')} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            {PRESETS.map(p => (
              <button key={p.label} type="button" className="btn btn-sm" onClick={() => setAdj({ ...NO_ADJUSTMENT, ...p.adj })}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <PixelCanvas image={image?.imageData ?? null} size={DISPLAY} pixelated={false} label="Original image" />
        </div>
        <div className="demo-panel">
          <h3>{heading}</h3>
          <PixelCanvas image={processed} size={DISPLAY} pixelated={false} label={heading} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Color Processing</h4>
        {view === 'rgb' && (
          <>
            <p><strong>RGB mode</strong> shows the full colour image with optional HSV adjustments. Use the sliders or preset buttons to transform colours.</p>
            {isDemo && (
              <p>The demo image has four quadrants: <span className="swatch-red">red</span>, <span className="swatch-green">green</span>, <span className="swatch-blue">blue</span>, <span className="swatch-yellow">yellow</span>, plus a pink centre circle.</p>
            )}
            <p><strong>Hue shift</strong> rotates colours around the wheel. <strong>Saturation</strong> controls colour intensity (0 = grayscale). <strong>Brightness</strong> scales V, the strongest channel.
            <strong> Gamma</strong> bends the tone curve (output = input<sup>1/γ</sup>): below 1 darkens mid-tones, above 1 lifts them, while pure black and white stay put.</p>
          </>
        )}
        {view === 'r' && <p><strong>Red channel only.</strong> Bright where red is present, dark elsewhere.</p>}
        {view === 'g' && <p><strong>Green channel only.</strong> Bright where green is present. The human eye is most sensitive to green.</p>}
        {view === 'b' && <p><strong>Blue channel only.</strong> Bright where blue is present. In camera photos the blue channel often carries the most sensor noise.</p>}
        {view === 'h' && <p><strong>Hue</strong> shows the pure colour of each pixel at full saturation. Shadows and highlights of the same colour share the same hue.</p>}
        {view === 's' && <p><strong>Saturation</strong> as a grayscale map. White = fully saturated (vivid colour), black = desaturated (gray).</p>}
        {view === 'v' && (
          <p><strong>Value</strong> is the <em>largest</em> of R, G and B, shown as a grayscale map.
          {isDemo
            ? ' Every colour in the demo image has one channel at 255, so the whole map is white — V ignores which channel is high. Perceived brightness (luma) would rank yellow above blue; upload a photo to see V capture shading.'
            : ' It captures shading and lighting but ignores hue: a pure blue and a pure yellow have the same V.'}
          </p>
        )}
      </div>
    </div>
  );
}
