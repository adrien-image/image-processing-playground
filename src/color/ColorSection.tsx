import { CodeViewer } from '../components/CodeViewer';
import colorCode from './ColorSection.tsx?raw';
import { useState, useRef, useEffect, useCallback } from 'react';
import { ImageUploader } from '../components/ImageUploader';

function generateColorImage(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 200; canvas.height = 200;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ff3333'; ctx.fillRect(0, 0, 100, 100);
  ctx.fillStyle = '#33ff33'; ctx.fillRect(100, 0, 100, 100);
  ctx.fillStyle = '#3333ff'; ctx.fillRect(0, 100, 100, 100);
  ctx.fillStyle = '#ffff33'; ctx.fillRect(100, 100, 100, 100);
  ctx.fillStyle = '#ff88ff'; ctx.beginPath(); ctx.arc(100, 100, 40, 0, Math.PI * 2); ctx.fill();
  return canvas.toDataURL();
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (mx === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  const s = mx === 0 ? 0 : d / mx;
  return [h, s, mx];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const [r1, g1, b1] = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return [r1 * 255, g1 * 255, b1 * 255];
}

type Channel = 'rgb' | 'r' | 'g' | 'b' | 'h' | 's' | 'v';

export function ColorSection() {
  const srcRef = useRef<HTMLImageElement>(null);
  const dstRef = useRef<HTMLCanvasElement>(null);
  const [channel, setChannel] = useState<Channel>('rgb');
  const [hueShift, setHueShift] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [satScale, setSatScale] = useState(1);
  const [valScale, setValScale] = useState(1);
  const [gamma, setGamma] = useState(1);

  const resetSliders = () => { setHueShift(0); setSatScale(1); setValScale(1); setGamma(1); };

  const handleUpload = useCallback((img: HTMLImageElement) => {
    if (srcRef.current) srcRef.current.src = img.src;
  }, []);

  useEffect(() => {
    if (srcRef.current) srcRef.current.src = generateColorImage();
  }, []);

  const process = useCallback(() => {
    const img = srcRef.current;
    const canvas = dstRef.current;
    if (!img || !canvas) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = img.naturalWidth || 200;
    canvas.height = img.naturalHeight || 200;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = imageData.data;

    if (channel === 'rgb') {
      // HSV adjustments
      if (hueShift !== 0 || satScale !== 1 || valScale !== 1 || gamma !== 1) {
        for (let i = 0; i < d.length; i += 4) {
          let [h, s, v] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
          h = (h + hueShift / 360) % 1;
          s = Math.min(1, s * satScale);
          v = Math.min(1, v * valScale);
          const [r1, g1, b1] = hsvToRgb(h, s, v);
          d[i] = r1; d[i + 1] = g1; d[i + 2] = b1;
        }
        // Gamma after HSV (on RGB)
        if (gamma !== 1) {
          const invGamma = 1 / gamma;
          for (let i = 0; i < d.length; i += 4) {
            d[i] = Math.pow(d[i] / 255, invGamma) * 255;
            d[i + 1] = Math.pow(d[i + 1] / 255, invGamma) * 255;
            d[i + 2] = Math.pow(d[i + 2] / 255, invGamma) * 255;
          }
        }
      }
    } else if (channel === 'r') {
      for (let i = 0; i < d.length; i += 4) { d[i + 1] = d[i]; d[i + 2] = d[i]; }
    } else if (channel === 'g') {
      for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 1]; d[i + 2] = d[i + 1]; }
    } else if (channel === 'b') {
      for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 2]; d[i + 1] = d[i + 2]; }
    } else if (channel === 'h') {
      for (let i = 0; i < d.length; i += 4) {
        const [h] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const [r1, g1, b1] = hsvToRgb(h, 1, 1);
        d[i] = r1; d[i + 1] = g1; d[i + 2] = b1;
      }
    } else if (channel === 's') {
      for (let i = 0; i < d.length; i += 4) {
        const [, s] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const v = Math.round(s * 255);
        d[i] = v; d[i + 1] = v; d[i + 2] = v;
      }
    } else if (channel === 'v') {
      for (let i = 0; i < d.length; i += 4) {
        const [, , v] = rgbToHsv(d[i], d[i + 1], d[i + 2]);
        const g = Math.round(v * 255);
        d[i] = g; d[i + 1] = g; d[i + 2] = g;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [channel, hueShift, satScale, valScale, gamma]);

  useEffect(() => { process(); }, [process]);

  const labelMap: Record<Channel, string> = { rgb: 'Processed', r: 'Red channel', g: 'Green channel', b: 'Blue channel', h: 'Hue', s: 'Saturation', v: 'Value' };

  return (
    <>
    <div className="color-section">
      <div className="demo-controls">
        <span className="demo-label">Channel:</span>
        {(['rgb', 'r', 'g', 'b', 'h', 's', 'v'] as Channel[]).map(c => (
          <button key={c} className={`btn btn-sm${channel === c ? ' btn-primary' : ''}`} onClick={() => setChannel(c)}>
            {c === 'rgb' ? 'RGB' : c === 'h' ? 'Hue' : c === 's' ? 'Sat' : c === 'v' ? 'Val' : c.toUpperCase()}
          </button>
        ))}
        <ImageUploader onImage={handleUpload} />
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: '0.5rem' }}>💻 Show code</button>
      </div>

      {channel === 'rgb' && (
        <div className="demo-slider-section">
          <div className="demo-slider-row">
            <span className="demo-slider-label">Hue shift: {hueShift}°</span>
            <input type="range" min={0} max={359} value={hueShift} onChange={e => setHueShift(Number(e.target.value))} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <span className="demo-slider-label">Saturation: {satScale.toFixed(2)}</span>
            <input type="range" min={0} max={2} step={0.05} value={satScale} onChange={e => setSatScale(Number(e.target.value))} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <span className="demo-slider-label">Brightness: {valScale.toFixed(2)}</span>
            <input type="range" min={0} max={2} step={0.05} value={valScale} onChange={e => setValScale(Number(e.target.value))} className="demo-slider" />
          </div>
          <div className="demo-slider-row">
            <span className="demo-slider-label">Gamma: {gamma.toFixed(2)}</span>
            <input type="range" min={0.1} max={3} step={0.05} value={gamma} onChange={e => setGamma(Number(e.target.value))} className="demo-slider" />
          </div>
          <div className="demo-slider-row" style={{ justifyContent: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
            <button className="btn btn-sm" onClick={resetSliders}>↺ Reset</button>
            <button className="btn btn-sm" onClick={() => { setSatScale(0); setValScale(1); setHueShift(0); }}>🖤 Desaturate</button>
            <button className="btn btn-sm" onClick={() => { setHueShift(180); setSatScale(1); setValScale(1); }}>🔄 Invert Hue</button>
            <button className="btn btn-sm" onClick={() => { setValScale(1.8); setSatScale(1); setHueShift(0); }}>☀ Overexpose</button>
            <button className="btn btn-sm" onClick={() => { setGamma(0.45); setSatScale(1); setValScale(1); setHueShift(0); }}>🌙 Darken</button>
          </div>
        </div>
      )}

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Original</h3>
          <img ref={srcRef} className="color-img" alt="original" />
        </div>
        <div className="demo-panel">
          <h3>{labelMap[channel]}</h3>
          <canvas ref={dstRef} className="color-img" />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Color Processing</h4>

        {channel === 'rgb' && (
          <>
            <p><strong>RGB mode</strong> shows the full colour image with optional HSV adjustments. Use the sliders or preset buttons to transform colours.</p>
            <p>The image has four quadrants: <span style={{color:'#ff3333'}}>red</span>, <span style={{color:'#33ff33'}}>green</span>, <span style={{color:'#3333ff'}}>blue</span>, <span style={{color:'#cccc33'}}>yellow</span>, plus a pink centre circle.</p>
            <p><strong>Hue shift</strong> rotates colours around the wheel. <strong>Saturation</strong> controls colour intensity (0 = grayscale). <strong>Brightness</strong> scales luminance.</p>
          </>
        )}

        {channel === 'r' && <p><strong>Red channel only.</strong> Bright where red is present, dark elsewhere.</p>}
        {channel === 'g' && <p><strong>Green channel only.</strong> Bright where green is present. The human eye is most sensitive to green.</p>}
        {channel === 'b' && <p><strong>Blue channel only.</strong> Bright where blue is present. The blue channel often carries the most sensor noise.</p>}
        {channel === 'h' && <p><strong>Hue</strong> shows the pure colour of each pixel at full saturation. Shadows and highlights of the same colour share the same hue.</p>}
        {channel === 's' && <p><strong>Saturation</strong> as a grayscale map. White = fully saturated (vivid colour), black = desaturated (gray).</p>}
        {channel === 'v' && <p><strong>Value (brightness)</strong> as a grayscale map. White = bright, black = dark. The yellow quadrant appears brightest (high R+G).</p>}
      </div>
    </div>
      {showCode && <CodeViewer code={colorCode} title="ColorSection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
