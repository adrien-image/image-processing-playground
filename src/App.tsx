import { Fragment, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { GROUPS, SECTIONS, SECTION_KEYS, groupText, sectionText, type SectionKey } from './app/sections';
import { useHashRoute } from './app/useHashRoute';
import { SectionErrorBoundary } from './app/SectionErrorBoundary';
import { Home } from './app/Home';
import { LessonNav } from './app/LessonNav';
import { searchSections } from './app/search';
import { SITE, SITE_TITLE } from './app/site';
import { LangSwitch } from './components/LangSwitch';
import { ThemeSwitch } from './components/ThemeSwitch';
import { useLang, useText } from './lib/i18n';

const ROUTES = ['home', ...SECTION_KEYS] as const;
type Route = (typeof ROUTES)[number];

const TEXT = {
  en: {
    overview: 'overview of all lessons',
    lessons: 'Lessons',
    allLessons: 'Overview of all lessons',
    search: 'Search lessons…',
    searchLabel: 'Search the lessons',
    noMatch: 'No lesson matches.',
    loading: (label: string) => `Loading ${label}…`,
    builtWith: <>Built with the <strong>Processor Registry</strong> pattern.</>,
    addFilter: 'Add your own convolution filter',
    steps: [
      <>1. Create <code>src/image-processing/processors/your-filter/index.ts</code> exporting a <code>Processor</code> (kernel, output mode, params).</>,
      <>2. Register it in <code>src/image-processing/processors/init.ts</code>.</>,
      <>3. It appears in the Convolution section's processor list, with controls generated from its params.</>,
    ],
  },
  fr: {
    overview: 'vue d\'ensemble des leçons',
    lessons: 'Leçons',
    allLessons: 'Vue d\'ensemble des leçons',
    search: 'Chercher une leçon…',
    searchLabel: 'Chercher dans les leçons',
    noMatch: 'Aucune leçon ne correspond.',
    loading: (label: string) => `Chargement de ${label}…`,
    builtWith: <>Construit avec le motif <strong>Processor Registry</strong> (registre de traitements).</>,
    addFilter: 'Ajouter votre propre filtre de convolution',
    steps: [
      <>1. Créez <code>src/image-processing/processors/your-filter/index.ts</code> qui exporte un <code>Processor</code> (noyau, mode de sortie, paramètres).</>,
      <>2. Enregistrez-le dans <code>src/image-processing/processors/init.ts</code>.</>,
      <>3. Il apparaît dans la liste des traitements de la leçon Convolution, avec des réglages générés à partir de ses paramètres.</>,
    ],
  },
};

function App() {
  const [route, navigate] = useHashRoute<Route>(ROUTES, 'home');
  const sectionKey: SectionKey | null = route === 'home' ? null : route;
  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState('');
  const anchorRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  // Typing right away searches: the search box takes focus when the menu opens.
  const focusSearch = useCallback((el: HTMLInputElement | null) => el?.focus(), []);
  const lang = useLang();
  const t = useText(TEXT);
  const site = useText(SITE);

  const section = sectionKey ? SECTIONS.find(s => s.key === sectionKey) ?? null : null;
  const sectionLabel = section ? sectionText(section, lang).label : null;
  const matches = searchSections(SECTIONS, query);

  useEffect(() => {
    document.title = sectionLabel ? `${sectionLabel} — ${SITE_TITLE}` : `${SITE_TITLE} — ${site.tagline}`;
  }, [sectionLabel, site]);

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
    setQuery('');
    window.scrollTo(0, 0);
  };

  return (
    <div className="app">
      <header className="app-header">
        <a
          href="#/home"
          className="header-brand"
          aria-label={`${SITE_TITLE}: ${t.overview}`}
          aria-current={route === 'home' ? 'page' : undefined}
          onClick={(e) => { e.preventDefault(); go('home'); }}
        >
          <img className="header-brand-logo" src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width={28} height={28} />
          <span className="header-brand-text">
            <span className="header-menu-title">{SITE_TITLE}</span>
            <span className="header-tagline">{site.tagline}</span>
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
                <span className="header-menu-current">{sectionLabel}</span>
              </>
            ) : (
              <span className="header-menu-current">{t.lessons}</span>
            )}
            <span className="header-menu-arrow" aria-hidden="true">▾</span>
          </button>

          {menuOpen && (
            <nav className="header-menu" id="section-menu" aria-label={t.lessons}>
              <input
                type="search"
                className="lesson-search header-menu-search"
                placeholder={t.search}
                aria-label={t.searchLabel}
                value={query}
                ref={focusSearch}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && query && matches[0]) go(matches[0].key); }}
              />
              <ul className="header-menu-grid">
                {!query && (
                  <li className="header-menu-home">
                    <a
                      href="#/home"
                      className={`header-menu-item${route === 'home' ? ' header-menu-item-active' : ''}`}
                      aria-current={route === 'home' ? 'page' : undefined}
                      onClick={(e) => { e.preventDefault(); go('home'); }}
                    >
                      <span className="header-menu-icon" aria-hidden="true">🏠</span>
                      <span className="header-menu-label">{t.allLessons}</span>
                    </a>
                  </li>
                )}
                {GROUPS.map(group => {
                  const inGroup = matches.filter(s => s.group === group.key);
                  if (!inGroup.length) return null;
                  const g = groupText(group, lang);
                  return (
                    <Fragment key={group.key}>
                      <li className="header-menu-group" aria-hidden="true">{group.icon} {g.label}</li>
                      {inGroup.map(s => (
                        <li key={s.key}>
                          <a
                            href={`#/${s.key}`}
                            className={`header-menu-item${s.key === sectionKey ? ' header-menu-item-active' : ''}`}
                            aria-current={s.key === sectionKey ? 'page' : undefined}
                            onClick={(e) => { e.preventDefault(); go(s.key); }}
                          >
                            <span className="header-menu-icon" aria-hidden="true">{s.icon}</span>
                            <span className="header-menu-label">{sectionText(s, lang).label}</span>
                          </a>
                        </li>
                      ))}
                    </Fragment>
                  );
                })}
                {!matches.length && <li className="header-menu-empty">{t.noMatch}</li>}
              </ul>
            </nav>
          )}
        </div>
        <ThemeSwitch />
        <LangSwitch />
        {menuOpen && <div className="header-menu-backdrop" aria-hidden="true" />}
      </header>

      <main className="section-container">
        {section ? (
          <>
            <SectionErrorBoundary key={section.key}>
              <Suspense fallback={<div className="section-loading" role="status">⏳ {t.loading(sectionLabel ?? '')}</div>}>
                <section.Component />
              </Suspense>
            </SectionErrorBoundary>
            <LessonNav current={section.key} onOpen={go} />
          </>
        ) : (
          <Home onOpen={go} />
        )}
      </main>

      <footer className="app-footer">
        <p>{t.builtWith}</p>
        <details className="footer-details">
          <summary>{t.addFilter}</summary>
          <div className="footer-details-content">
            {t.steps.map((step, i) => <p key={i}>{step}</p>)}
          </div>
        </details>
      </footer>
    </div>
  );
}

export default App;
