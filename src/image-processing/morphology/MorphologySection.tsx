import { useMemo, useState } from 'react';
import morphCode from './morphology.ts?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { PixelCanvas } from '../../components/PixelCanvas';
import { ImageUploader } from '../../components/ImageUploader';
import { useImageData, type ImageSource } from '../../hooks/useImageData';
import { makeCanvas } from '../../lib/demoImages';
import { grayToImageData } from '../../lib/image';
import { histogram256, otsuThreshold } from '../segmentation/otsu';
import { binarize, morphology, type MorphOp } from './morphology';

/** Shapes plus small white specks (removed by opening) and black holes (filled by closing). */
function binaryDemo(): HTMLCanvasElement {
  return makeCanvas(100, 100, ctx => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 100, 100);
    ctx.fillStyle = '#fff';
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
    // Small holes
    ctx.fillStyle = '#000';
    ctx.fillRect(25, 25, 3, 3);
    ctx.fillRect(38, 18, 2, 2);
    ctx.fillRect(72, 70, 3, 3);
  });
}

/** Gray shapes on a gradient with bright and dark speckle — grayscale morphology acts on every level. */
function grayDemo(): HTMLCanvasElement {
  return makeCanvas(100, 100, ctx => {
    const g = ctx.createLinearGradient(0, 0, 100, 100);
    g.addColorStop(0, '#202020');
    g.addColorStop(1, '#707070');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 100, 100);
    ctx.fillStyle = '#c8c8c8'; ctx.fillRect(10, 10, 38, 38);
    ctx.fillStyle = '#9a9a9a'; ctx.beginPath(); ctx.arc(72, 70, 20, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e8e8e8'; ctx.fillRect(58, 12, 28, 12);
    // Deterministic speckle: bright dots on dark areas, dark dots inside the bright shapes.
    for (let i = 0; i < 60; i++) {
      const x = (i * 37) % 97, y = (i * 61) % 89;
      ctx.fillStyle = i % 2 ? '#ffffff' : '#000000';
      ctx.fillRect(x, y, 2, 2);
    }
  });
}

const OPS: { key: MorphOp; label: string; heading: string }[] = [
  { key: 'erode', label: 'Erosion', heading: 'Eroded' },
  { key: 'dilate', label: 'Dilation', heading: 'Dilated' },
  { key: 'open', label: 'Opening', heading: 'Opening' },
  { key: 'close', label: 'Closing', heading: 'Closing' },
];

const DISPLAY = 260;

export function MorphologySection() {
  const [op, setOp] = useState<MorphOp>('erode');
  const [kernelSize, setKernelSize] = useState(3);
  const [showCode, setShowCode] = useState(false);
  const [grayMode, setGrayMode] = useState(false);
  const [upload, setUpload] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Each mode has its own demo; an upload replaces both until cleared.
  const demo = useMemo<ImageSource>(() => (grayMode ? grayDemo() : binaryDemo()), [grayMode]);
  const { image, error: loadError } = useImageData(upload ?? demo, 256);

  const input = useMemo(() => {
    if (!image) return null;
    if (grayMode) return image.gray;
    // Binary mode: threshold with Otsu so uploaded photos binarise sensibly.
    return binarize(image.gray, otsuThreshold(histogram256(image.gray)));
  }, [image, grayMode]);

  const result = useMemo(
    () => (image && input ? morphology(input, image.width, image.height, op, kernelSize) : null),
    [image, input, op, kernelSize],
  );

  const inputImage = useMemo(() => (input && image ? grayToImageData(input, image.width, image.height) : null), [input, image]);
  const resultImage = useMemo(() => (result && image ? grayToImageData(result, image.width, image.height) : null), [result, image]);

  const error = uploadError ?? loadError;
  const opInfo = OPS.find(o => o.key === op)!;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Operation:</span>
        {OPS.map(o => (
          <button key={o.key} type="button" aria-pressed={op === o.key}
            className={`btn btn-sm${op === o.key ? ' btn-primary' : ''}`} onClick={() => setOp(o.key)}>
            {o.label}
          </button>
        ))}
        <span className="demo-label">Mode:</span>
        <button type="button" aria-pressed={!grayMode} className={`btn btn-sm${!grayMode ? ' btn-primary' : ''}`} onClick={() => setGrayMode(false)}>Binary</button>
        <button type="button" aria-pressed={grayMode} className={`btn btn-sm${grayMode ? ' btn-primary' : ''}`} onClick={() => setGrayMode(true)}>Grayscale</button>
        <ImageUploader label="📁 Upload" onImage={(src) => { setUpload(src); setUploadError(null); }} onError={setUploadError} />
        {upload && <button type="button" className="btn btn-sm" onClick={() => setUpload(null)}>Demo image</button>}
        <span className="ml-auto">
          <ShowCodeButton code={morphCode} title="morphology/morphology.ts" open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      {error && <div className="error-banner" role="alert">⚠ {error}</div>}

      <div className="demo-slider-section">
        <div className="demo-slider-row">
          <label className="demo-slider-label" htmlFor="morph-size">Kernel size: <strong>{kernelSize}×{kernelSize}</strong></label>
          <input id="morph-size" type="range" min={3} max={15} step={2} value={kernelSize}
            onChange={e => setKernelSize(Number(e.target.value))} className="demo-slider" />
        </div>
      </div>

      <div className="demo-images">
        <div className="demo-panel">
          <h3>{grayMode ? 'Grayscale input' : 'Binary input'}</h3>
          <PixelCanvas image={inputImage} size={DISPLAY} label="Input image" />
        </div>
        <div className="demo-panel">
          <h3>{opInfo.heading}</h3>
          <PixelCanvas image={resultImage} size={DISPLAY} label={`${opInfo.heading} image`} />
        </div>
      </div>

      <div className="demo-explanation">
        <h4>Morphological Operations</h4>
        {grayMode ? (
          <>
            <p><strong>Erosion</strong> replaces each pixel by the <em>darkest</em> value in its k×k neighbourhood: bright shapes shrink and bright specks vanish, dark specks grow.</p>
            <p><strong>Dilation</strong> takes the <em>brightest</em> value: bright shapes grow, dark specks inside them vanish.</p>
            <p><strong>Opening</strong> (erosion → dilation) removes bright details smaller than the kernel while keeping larger shapes and the background gradient.
            <strong> Closing</strong> (dilation → erosion) removes small dark details the same way.</p>
          </>
        ) : (
          <>
            <p><strong>Erosion:</strong> shrinks white regions. A pixel stays white only if <em>every</em> pixel in the kernel neighbourhood is white. Removes small protrusions and noise specks.</p>
            <p><strong>Dilation:</strong> expands white regions. A pixel becomes white if <em>any</em> pixel in the kernel neighbourhood is white. Fills small holes and gaps.</p>
            <p><strong>Opening</strong> (erosion → dilation): removes the small white specks while preserving the size of the big shapes.</p>
            <p><strong>Closing</strong> (dilation → erosion): fills the small black holes while preserving shape size.</p>
          </>
        )}
      </div>
    </div>
  );
}
