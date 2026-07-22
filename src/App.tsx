import { useState, useEffect, useCallback, useRef } from 'react';
import { ConvolutionSection } from './convolution/ConvolutionSection';
import { DLPlayground } from './dl/DLPlayground';
import { SegmentationSection } from './segmentation/SegmentationSection';
import { MatchingSection } from './matching/MatchingSection';
import { ColorSection } from './color/ColorSection';
import { MorphologySection } from './morphology/MorphologySection';
import { EdgeComparisonSection } from './edges/EdgeComparisonSection';
import { FrequencySection } from './frequency/FrequencySection';
import { HoughSection } from './hough/HoughSection';
import { HistEqSection } from './histogram/HistEqSection';
import { AugmentationSection } from './augmentation/AugmentationSection';
import { DenoisingSection } from './denoising/DenoisingSection';
import { TemplateSection } from './template/TemplateSection';
import { initProcessors } from './processors';
import './App.css';

type Section = 'convolution' | 'deep-learning' | 'segmentation' | 'matching' | 'color' | 'morphology' | 'edges' | 'frequency' | 'hough' | 'histeq' | 'augmentation' | 'denoising' | 'template';

const SECTIONS: { key: Section; icon: string; label: string }[] = [
  { key: 'convolution',   icon: '🖼', label: 'Convolution' },
  { key: 'segmentation',  icon: '🎯', label: 'Segmentation' },
  { key: 'matching',      icon: '🔗', label: 'Matching' },
  { key: 'color',         icon: '🎨', label: 'Color' },
  { key: 'morphology',    icon: '🔲', label: 'Morphology' },
  { key: 'edges',         icon: '⚡', label: 'Edges' },
  { key: 'frequency',     icon: '🌀', label: 'Frequency' },
  { key: 'hough',         icon: '📐', label: 'Hough' },
  { key: 'histeq',        icon: '📊', label: 'Histogram' },
  { key: 'augmentation',  icon: '📦', label: 'Augmentation' },
  { key: 'denoising',     icon: '🧹', label: 'Denoising' },
  { key: 'template',      icon: '🔍', label: 'Template' },
  { key: 'deep-learning', icon: '🧠', label: 'Classification' },
];

function App() {
  const [section, setSection] = useState<Section>('convolution');
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const currentSection = SECTIONS.find(s => s.key === section)!;

  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') setMenuOpen(false);
  }, []);

  useEffect(() => {
    initProcessors();
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [handleKey]);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [menuOpen]);

  const navigate = (key: Section) => { setSection(key); setMenuOpen(false); };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-menu-anchor">
          <button className="header-menu-btn" onClick={() => setMenuOpen(!menuOpen)}>
            {currentSection.icon}{' '}
            <span className="header-menu-title">Image Processing Playground</span>
            <span className="header-menu-arrow">▾</span>
          </button>

          {menuOpen && (
            <>
              <div className="header-menu-backdrop" onClick={() => setMenuOpen(false)} />
              <div className="header-menu" ref={menuRef}>
                <div className="header-menu-grid">
                  {SECTIONS.map(s => (
                    <button key={s.key}
                      className={`header-menu-item${section === s.key ? ' header-menu-item-active' : ''}`}
                      onClick={() => navigate(s.key)}>
                      <span className="header-menu-icon">{s.icon}</span>
                      <span className="header-menu-label">{s.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

      </header>

      {section === 'convolution' && <ConvolutionSection />}
      {section === 'deep-learning' && <div className="dl-container"><DLPlayground /></div>}
      {section === 'segmentation' && <div className="seg-container"><SegmentationSection /></div>}
      {section === 'matching' && <div className="matching-container"><MatchingSection /></div>}
      {section === 'color' && <div className="seg-container"><ColorSection /></div>}
      {section === 'morphology' && <div className="seg-container"><MorphologySection /></div>}
      {section === 'edges' && <div className="seg-container"><EdgeComparisonSection /></div>}
      {section === 'frequency' && <div className="seg-container"><FrequencySection /></div>}
      {section === 'hough' && <div className="seg-container"><HoughSection /></div>}
      {section === 'histeq' && <div className="seg-container"><HistEqSection /></div>}
      {section === 'augmentation' && <div className="seg-container"><AugmentationSection /></div>}
      {section === 'denoising' && <div className="seg-container"><DenoisingSection /></div>}
      {section === 'template' && <div className="seg-container"><TemplateSection /></div>}

      <footer className="app-footer">
        <p>Built with the <strong>Processor Registry</strong> pattern —
          <details className="footer-details">
            <summary>add your own filter</summary>
            <div className="footer-details-content">
              <p>1. Create a file in <code>src/processors/your-filter/index.ts</code> exporting a <code>Processor</code> object.</p>
              <p>2. Import and register it in <code>src/processors/init.ts</code>.</p>
              <p>3. The UI discovers it automatically.</p>
            </div>
          </details>.</p>
      </footer>
    </div>
  );
}

export default App;
