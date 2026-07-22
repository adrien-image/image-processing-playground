import { CodeViewer } from '../components/CodeViewer';
import convolveCode from '../processors/utils.ts?raw';
import { useState, useCallback, useEffect, useRef } from 'react';
import { ImageSourceSelector } from '../components/ImageSourceSelector';
import { ProcessorSelector } from '../components/ProcessorSelector';
import { ImageDisplay } from '../components/ImageDisplay';
import { ProgressiveProcessing, MathPanel, type MathPanelData } from '../components/ConvolutionAnimation';
import { KernelMatrixDisplay } from '../components/KernelMatrixDisplay';
import { loadImage, getImageData, generateCheckerboardDataURL, generateCircleDataURL, generateEdgesDataURL, generateCornerDataURL } from '../utils/imageLoader';
import { SOBEL_X, SOBEL_Y } from '../processors/sobel/kernels';
import { GAUSSIAN_3x3 } from '../processors/gaussian-blur';
import { SHARPEN_3x3 } from '../processors/sharpen';

type ProcMode = 'absolute' | 'direct';

interface ProcConfig {
  kernel: number[][];
  mode: ProcMode;
  label: string;
}

const PROC_CONFIGS: Record<string, ProcConfig> = {
  'sobel':         { kernel: SOBEL_X,     mode: 'absolute', label: 'Sobel Gx — Vertical Edges' },
  'sobel-gy':      { kernel: SOBEL_Y,     mode: 'absolute', label: 'Sobel Gy — Horizontal Edges' },
  'gaussian-blur': { kernel: GAUSSIAN_3x3, mode: 'direct',  label: 'Gaussian Blur' },
  'sharpen':       { kernel: SHARPEN_3x3,  mode: 'direct',  label: 'Sharpen' },
};

type IpTab = 'processing' | 'computation';

export function ConvolutionSection() {
  const [ipTab, setIpTab] = useState<IpTab>('processing');
  const [originalData, setOriginalData] = useState<ImageData | null>(null);
  const [partialResult, setPartialResult] = useState<ImageData | null>(null);
  const [width, setWidth] = useState(0);
  const [showCode, setShowCode] = useState(false);
  const [height, setHeight] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeProcessor, setActiveProcessor] = useState<string | null>(null);
  const [highlightPosition, setHighlightPosition] = useState<{ col: number; row: number } | null>(null);
  const [mathData, setMathData] = useState<MathPanelData | null>(null);

  const loadDemoImage = useCallback(async (name: 'checkerboard' | 'circle' | 'edges' | 'corner') => {
    setIsProcessing(true);
    setError(null);
    setPartialResult(null);
    setActiveProcessor(null);
    setHighlightPosition(null);
    setMathData(null);

    const generators: Record<string, () => string> = {
      checkerboard: generateCheckerboardDataURL,
      circle: generateCircleDataURL,
      edges: generateEdgesDataURL,
      corner: generateCornerDataURL,
    };

    try {
      const img = await loadImage(generators[name]());
      const { imageData, width: w, height: h } = getImageData(img, 64);
      setOriginalData(imageData);
      setWidth(w);
      setHeight(h);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load image');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleProcessorSelect = useCallback((id: string) => {
    setActiveProcessor(id);
    setPartialResult(null);
    setMathData(null);
  }, []);

  const handlePartialResult = useCallback((data: ImageData) => {
    setPartialResult(data);
  }, []);

  const handleStep = useCallback((col: number, row: number) => {
    setHighlightPosition({ col, row });
  }, []);

  const handleMathData = useCallback((data: MathPanelData) => {
    setMathData(data);
  }, []);

  const initialLoad = useRef(true);
  useEffect(() => {
    if (initialLoad.current) {
      initialLoad.current = false;
      loadDemoImage('checkerboard');
    }
  }, [loadDemoImage]);

  const procConfig = activeProcessor ? PROC_CONFIGS[activeProcessor] : null;

  return (
    <>
    <div className="app-layout">
      <aside className="sidebar">
        <ImageSourceSelector onSelect={loadDemoImage} disabled={isProcessing} />
        <ProcessorSelector
          onSelect={handleProcessorSelect}
          activeProcessor={activeProcessor}
          disabled={isProcessing || !originalData}
          config={{}}
          onConfigChange={() => {}}
        />
        {procConfig && <KernelMatrixDisplay kernel={procConfig.kernel} label={procConfig.label} />}
        {error && <div className="error-banner">⚠ {error}</div>}
        {isProcessing && <div className="loading">⏳ Loading…</div>}
      </aside>

      <main className="main-content">
        <div className="tab-bar">
          <button className={`tab-btn${ipTab === 'processing' ? ' tab-active' : ''}`}
            onClick={() => setIpTab('processing')}>Processing</button>
          <button className={`tab-btn${ipTab === 'computation' ? ' tab-active' : ''}`}
            onClick={() => setIpTab('computation')}>Computation</button>
        <button className="btn btn-sm" onClick={() => setShowCode(true)} style={{ marginLeft: "0.5rem" }}>💻 Show code</button>
        </div>

        {ipTab === 'processing' && (
          <>
            <ImageDisplay original={originalData} processed={partialResult} width={width} height={height}
              highlightPosition={activeProcessor ? highlightPosition : null} />
            {activeProcessor && originalData && procConfig && (
              <ProgressiveProcessing imageData={originalData} active={true} kernel={procConfig.kernel}
                mode={procConfig.mode} onPartialResult={handlePartialResult} onStep={handleStep}
                onMathData={handleMathData} />
            )}
          </>
        )}

        {ipTab === 'computation' && (
          <div className="tab-content-computation">
            {mathData ? <MathPanel data={mathData} /> : (
              <div className="placeholder" style={{ padding: '3rem', textAlign: 'center' }}>
                <span>Select a processor and start the animation to see the computation details.</span>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
      {showCode && <CodeViewer code={convolveCode} title="processors/utils.ts" onClose={() => setShowCode(false)} />}
    </>
  );
}
