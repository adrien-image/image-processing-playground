import { Fragment, Suspense, useEffect, useRef, useState } from 'react';
import { GROUPS, SECTIONS, SECTION_KEYS, sectionsIn, type SectionKey } from './app/sections';
import { useHashRoute } from './app/useHashRoute';
import { SectionErrorBoundary } from './app/SectionErrorBoundary';
import { Home } from './app/Home';
import { SITE_TAGLINE, SITE_TITLE } from './app/site';
import { ThemeSwitch } from './components/ThemeSwitch';

const ROUTES = ['home', ...SECTION_KEYS] as const;
type Route = (typeof ROUTES)[number];

function App() {
  const [route, navigate] = useHashRoute<Route>(ROUTES, 'home');
  const sectionKey: SectionKey | null = route === 'home' ? null : route;
  const [menuOpen, setMenuOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const section = sectionKey ? SECTIONS.find(s => s.key === sectionKey) ?? null : null;

  useEffect(() => {
    document.title = section ? `${section.label} — ${SITE_TITLE}` : `${SITE_TITLE} — ${SITE_TAGLINE}`;
  }, [section]);

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

  const go = (key: Route) => {
    navigate(key);
    setMenuOpen(false);
    window.scrollTo(0, 0);
  };

  return (
    <div className="app">
      <header className="app-header">
        <a
          href="#/home"
          className="header-brand"
          aria-label={`${SITE_TITLE}: overview of all lessons`}
          aria-current={route === 'home' ? 'page' : undefined}
          onClick={(e) => { e.preventDefault(); go('home'); }}
        >
          <img className="header-brand-logo" src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width={28} height={28} />
          <span className="header-brand-text">
            <span className="header-menu-title">{SITE_TITLE}</span>
            <span className="header-tagline">{SITE_TAGLINE}</span>
          </span>
        </a>

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
            {section ? (
              <>
                <span aria-hidden="true">{section.icon}</span>
                <span className="header-menu-current">{section.label}</span>
              </>
            ) : (
              <span className="header-menu-current">Lessons</span>
            )}
            <span className="header-menu-arrow" aria-hidden="true">▾</span>
          </button>

          {menuOpen && (
            <nav className="header-menu" id="section-menu" aria-label="Lessons">
              <ul className="header-menu-grid">
                <li className="header-menu-home">
                  <a
                    href="#/home"
                    className={`header-menu-item${route === 'home' ? ' header-menu-item-active' : ''}`}
                    aria-current={route === 'home' ? 'page' : undefined}
                    onClick={(e) => { e.preventDefault(); go('home'); }}
                  >
                    <span className="header-menu-icon" aria-hidden="true">🏠</span>
                    <span className="header-menu-label">Overview of all lessons</span>
                  </a>
                </li>
                {GROUPS.map(group => (
                  <Fragment key={group.key}>
                    <li className="header-menu-group" aria-hidden="true">{group.icon} {group.label}</li>
                    {sectionsIn(group.key).map(s => (
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
                  </Fragment>
                ))}
              </ul>
            </nav>
          )}
        </div>
        <ThemeSwitch />
        {menuOpen && <div className="header-menu-backdrop" aria-hidden="true" />}
      </header>

      <main className="section-container">
        {section ? (
          <SectionErrorBoundary key={section.key}>
            <Suspense fallback={<div className="section-loading" role="status">⏳ Loading {section.label}…</div>}>
              <section.Component />
            </Suspense>
          </SectionErrorBoundary>
        ) : (
          <Home onOpen={go} />
        )}
      </main>

      <footer className="app-footer">
        <p>Built with the <strong>Processor Registry</strong> pattern.</p>
        <details className="footer-details">
          <summary>Add your own convolution filter</summary>
          <div className="footer-details-content">
            <p>1. Create <code>src/image-processing/processors/your-filter/index.ts</code> exporting a <code>Processor</code> (kernel, output mode, params).</p>
            <p>2. Register it in <code>src/image-processing/processors/init.ts</code>.</p>
            <p>3. It appears in the Convolution section's processor list, with controls generated from its params.</p>
          </div>
        </details>
      </footer>
    </div>
  );
}

export default App;
