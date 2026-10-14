import { Suspense, useEffect, useRef, useState } from 'react';
import { SECTIONS, SECTION_KEYS, type SectionKey } from './app/sections';
import { useHashRoute } from './app/useHashRoute';
import { SectionErrorBoundary } from './app/SectionErrorBoundary';

function App() {
  const [sectionKey, navigate] = useHashRoute<SectionKey>(SECTION_KEYS, 'convolution');
  const [menuOpen, setMenuOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const section = SECTIONS.find(s => s.key === sectionKey) ?? SECTIONS[0];
  const { Component } = section;

  useEffect(() => {
    document.title = `${section.label} — Image Processing Playground`;
  }, [section.label]);

  // Close on Escape (returning focus to the menu button) or on a click outside the
  // menu. The toggle button is inside `anchorRef`, so clicking it still toggles.
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setMenuOpen(false);
        buttonRef.current?.focus();
      }
    };
    const onMouseDown = (e: MouseEvent) => {
      if (anchorRef.current && !anchorRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onMouseDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onMouseDown);
    };
  }, [menuOpen]);

  const go = (key: SectionKey) => {
    navigate(key);
    setMenuOpen(false);
    buttonRef.current?.focus();
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-menu-anchor" ref={anchorRef}>
          <button
            ref={buttonRef}
            type="button"
            className="header-menu-btn"
            aria-haspopup="true"
            aria-expanded={menuOpen}
            aria-controls="section-menu"
            onClick={() => setMenuOpen(open => !open)}
          >
            <span aria-hidden="true">{section.icon}</span>
            <span className="header-menu-title">Image Processing Playground</span>
            <span className="header-menu-current">· {section.label}</span>
            <span className="header-menu-arrow" aria-hidden="true">▾</span>
          </button>

          {menuOpen && (
            <nav className="header-menu" id="section-menu" aria-label="Sections">
              <ul className="header-menu-grid">
                {SECTIONS.map(s => (
                  <li key={s.key}>
                    <a
                      href={`#/${s.key}`}
                      className={`header-menu-item${s.key === sectionKey ? ' header-menu-item-active' : ''}`}
                      aria-current={s.key === sectionKey ? 'page' : undefined}
                      onClick={(e) => { e.preventDefault(); go(s.key); }}
                    >
                      <span className="header-menu-icon" aria-hidden="true">{s.icon}</span>
                      <span className="header-menu-label">{s.label}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          )}
        </div>
        {menuOpen && <div className="header-menu-backdrop" aria-hidden="true" />}
      </header>

      <main className="section-container">
        <SectionErrorBoundary key={sectionKey}>
          <Suspense fallback={<div className="section-loading" role="status">⏳ Loading {section.label}…</div>}>
            <Component />
          </Suspense>
        </SectionErrorBoundary>
      </main>

      <footer className="app-footer">
        <p>Built with the <strong>Processor Registry</strong> pattern.</p>
        <details className="footer-details">
          <summary>Add your own convolution filter</summary>
          <div className="footer-details-content">
            <p>1. Create <code>src/processors/your-filter/index.ts</code> exporting a <code>Processor</code> (kernel, output mode, params).</p>
            <p>2. Register it in <code>src/processors/init.ts</code>.</p>
            <p>3. It appears in the Convolution section's processor list, with controls generated from its params.</p>
          </div>
        </details>
      </footer>
    </div>
  );
}

export default App;
