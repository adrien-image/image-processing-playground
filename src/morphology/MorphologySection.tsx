import { CodeViewer } from '../components/CodeViewer';
import morphCode from './MorphologySection.tsx?raw';
import { useState, useRef, useEffect, useCallback } from 'react';
import { rgbaToGrayscale, grayscaleToRGBA } from '../processors/utils';

// Generate a binary demo image with shapes
function generateBinaryImage(): string {
  const canvas = document.createElement('canvas');
  canvas.width = 100; canvas.height = 100;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 100, 100);
  ctx.fillStyle = '#fff';
  // A rectangle, a circle, some noise dots
  ctx.fillRect(10, 10, 40, 40);
  ctx.beginPath(); ctx.arc(75, 75, 20, 0, Math.PI * 2); ctx.fill();
  ctx.fillRect(50, 15, 10, 10);
  ctx.fillRect(15, 60, 15, 15);
  // Noise specks
  ctx.fillRect(5, 50, 3, 3);
  ctx.fillRect(60, 10, 3, 3);
  ctx.fillRect(80, 40, 2, 2);
  ctx.fillRect(45, 80, 2, 2);
  ctx.fillRect(90, 85, 3, 3);
  return canvas.toDataURL();
}

type MorphOp = 'erode' | 'dilate' | 'open' | 'close';

function morphBinary(pixels: Float64Array, width: number, height: number, op: MorphOp, kernelSize: number): Float64Array {
  const result = new Float64Array(pixels.length);
  const half = Math.floor(kernelSize / 2);

  const apply = (input: Float64Array, out: Float64Array, erode: boolean) => {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let best = erode ? 255 : 0;
        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            const px = x + kx, py = y + ky;
            if (px >= 0 && px < width && py >= 0 && py < height) {
              const v = input[py * width + px];
              if (erode) best = Math.min(best, v);
              else best = Math.max(best, v);
            }
          }
        }
        out[y * width + x] = best;
      }
    }
  };

  if (op === 'erode') apply(pixels, result, true);
  else if (op === 'dilate') apply(pixels, result, false);
  else if (op === 'open') {
    const temp = new Float64Array(pixels.length);
    apply(pixels, temp, true);    // erode
    apply(temp, result, false);    // dilate
  } else if (op === 'close') {
    const temp = new Float64Array(pixels.length);
    apply(pixels, temp, false);   // dilate
    apply(temp, result, true);    // erode
  }

  return result;
}

export function MorphologySection() {
  const [op, setOp] = useState<MorphOp>('erode');
  const [kernelSize, setKernelSize] = useState(3);
  const [showCode, setShowCode] = useState(false);
  const [grayMode, setGrayMode] = useState(false);
  const srcRef = useRef<HTMLImageElement>(null);
  const dstRef = useRef<HTMLCanvasElement>(null);

  const process = useCallback(() => {
    const img = srcRef.current;
    const canvas = dstRef.current;
    if (!img || !canvas) return;
    if (!img.complete || img.naturalWidth === 0) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = img.naturalWidth || 100;
    tempCanvas.height = img.naturalHeight || 100;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.drawImage(img, 0, 0);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const gray = rgbaToGrayscale(imageData.data, tempCanvas.width, tempCanvas.height);
    if (!grayMode) {
      for (let i = 0; i < gray.length; i++) gray[i] = gray[i] > 128 ? 255 : 0;
    }
    const result = morphBinary(gray, tempCanvas.width, tempCanvas.height, op, kernelSize);
    canvas.width = tempCanvas.width;
    canvas.height = tempCanvas.height;
    const ctx = canvas.getContext('2d')!;
    const rgba = grayscaleToRGBA(result, tempCanvas.width, tempCanvas.height);
    ctx.putImageData(new ImageData(rgba, tempCanvas.width, tempCanvas.height), 0, 0);
  }, [op, kernelSize, grayMode]);

  useEffect(() => {
    const img = srcRef.current;
    if (!img) return;
    const onLoad = () => process();
    img.onload = onLoad;
    img.src = generateBinaryImage();
    if (img.complete && img.naturalWidth) process();
    return () => { img.onload = null; };
  }, [process]);

  return (
    <>
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Operation:</span>
        <button className={`btn btn-sm${op === 'erode' ? ' btn-primary' : ''}`} onClick={() => setOp('erode')}>Erosion</button>
        <button className={`btn btn-sm${op === 'dilate' ? ' btn-primary' : ''}`} onClick={() => setOp('dilate')}>Dilation</button>
        <button className={`btn btn-sm${op === 'open' ? ' btn-primary' : ''}`} onClick={() => setOp('open')}>Opening</button>
        <button className={`btn btn-sm${op === 'close' ? ' btn-primary' : ''}`} onClick={() => setOp('close')}>Closing</button>
        <span className="demo-label" style={{ marginLeft: '1rem' }}>Mode:</span>
        <button className={`btn btn-sm${!grayMode ? ' btn-primary' : ''}`} onClick={() => setGrayMode(false)}>Binary</button>
        <button className={`btn btn-sm${grayMode ? ' btn-primary' : ''}`} onClick={() => setGrayMode(true)}>Grayscale</button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: "0.5rem" }}>💻 Show code</button>
      </div>

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <span className="demo-slider-label">Kernel size: <strong>{kernelSize}×{kernelSize}</strong></span>
          <input type="range" min={3} max={15} step={2} value={kernelSize}
            onChange={e => setKernelSize(Number(e.target.value))} className="demo-slider" />
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>Binary input</h3>
          <img ref={srcRef} className="color-img" alt="input" />
        </div>
        <div className="demo-panel">
          <h3>{op === 'erode' ? 'Eroded' : op === 'dilate' ? 'Dilated' : op === 'open' ? 'Opening' : 'Closing'}</h3>
          <canvas ref={dstRef} className="color-img" />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Morphological Operations</h4>
        <p><strong>Erosion:</strong> shrinks white regions. A pixel is white only if <em>every</em> pixel in the kernel neighbourhood is white. Removes small protrusions and noise specks.</p>
        <p><strong>Dilation:</strong> expands white regions. A pixel is white if <em>any</em> pixel in the kernel neighbourhood is white. Fills small holes and gaps.</p>
        <p><strong>Opening</strong> (erosion → dilation): removes small white noise while preserving shape size.</p>
        <p><strong>Closing</strong> (dilation → erosion): fills small black holes while preserving shape size.</p>
      </div>
    </div>
      {showCode && <CodeViewer code={morphCode} title="MorphologySection.tsx" onClose={() => setShowCode(false)} />}
    </>
  );
}
