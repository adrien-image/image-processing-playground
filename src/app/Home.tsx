import { useState } from 'react';
import { useLang, useText } from '../lib/i18n';
import { searchSections } from './search';
import { GROUPS, SECTIONS, groupText, sectionText, type SectionKey } from './sections';
import { SITE, SITE_TITLE } from './site';
import './home.css';

interface HomeProps {
  onOpen: (key: SectionKey) => void;
}

const TEXT = {
  en: {
    surprise: 'Surprise me',
    count: (n: number) => `${n} lessons · runs entirely in your browser · bring your own images`,
    lessons: (n: number) => `${n} lesson${n === 1 ? '' : 's'}`,
    search: 'Search the lessons: “edges”, “svm”, “gradient”…',
    searchLabel: 'Search the lessons',
    found: (n: number) => (n === 1 ? '1 lesson found' : `${n} lessons found`),
    none: 'No lesson matches. Try fewer or shorter words.',
  },
  fr: {
    surprise: 'Surprenez-moi',
    count: (n: number) => `${n} leçons · tout tourne dans votre navigateur · apportez vos propres images`,
    lessons: (n: number) => `${n} leçon${n === 1 ? '' : 's'}`,
    search: 'Chercher une leçon : « contours », « svm », « gradient »…',
    searchLabel: 'Chercher dans les leçons',
    found: (n: number) => (n === 1 ? '1 leçon trouvée' : `${n} leçons trouvées`),
    none: 'Aucune leçon ne correspond. Essayez des mots moins nombreux ou plus courts.',
  },
};

/** Landing page: what the playground is, a search box, and every lesson as a card. */
export function Home({ onOpen }: HomeProps) {
  const lang = useLang();
  const t = useText(TEXT);
  const site = useText(SITE);
  const [query, setQuery] = useState('');
  const matches = searchSections(SECTIONS, query);

  return (
    <div className="home">
      <section className="home-hero" aria-labelledby="home-title">
        <h1 id="home-title" className="home-title">{SITE_TITLE}</h1>
        <p className="home-tagline">{site.tagline}</p>
        <p className="home-intro">{site.description}</p>
        <div className="home-actions">
          {GROUPS.map((group, i) => (
            <a
              key={group.key}
              href={`#/${group.start}`}
              className={`btn home-cta${i === 0 ? ' btn-primary' : ''}`}
              onClick={(e) => { e.preventDefault(); onOpen(group.start); }}
            >
              {groupText(group, lang).cta} →
            </a>
          ))}
          <button type="button" className="btn home-cta" onClick={() => onOpen(SECTIONS[Math.floor(Math.random() * SECTIONS.length)].key)}>
            🎲 {t.surprise}
          </button>
        </div>
        <p className="home-count">{t.count(SECTIONS.length)}</p>
      </section>

      <div className="home-search">
        <input
          type="search"
          className="lesson-search"
          placeholder={t.search}
          aria-label={t.searchLabel}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && query && matches[0]) onOpen(matches[0].key); }}
        />
        {query && <p className="home-search-status" role="status">{matches.length ? t.found(matches.length) : t.none}</p>}
      </div>

      {GROUPS.map(group => {
        const inGroup = matches.filter(s => s.group === group.key);
        if (!inGroup.length) return null;
        const g = groupText(group, lang);
        return (
          <section key={group.key} className="home-group" aria-labelledby={`home-group-${group.key}`}>
            <header className="home-group-head">
              <h2 id={`home-group-${group.key}`} className="home-group-title">
                <span aria-hidden="true">{group.icon}</span> {g.label}
                <span className="home-group-count">{t.lessons(inGroup.length)}</span>
              </h2>
              <p className="home-group-blurb">{g.blurb}</p>
            </header>
            <ol className="home-grid">
              {inGroup.map(s => {
                const st = sectionText(s, lang);
                const number = SECTIONS.filter(o => o.group === group.key).indexOf(s) + 1;
                return (
                  <li key={s.key}>
                    <a
                      href={`#/${s.key}`}
                      className="home-card"
                      onClick={(e) => { e.preventDefault(); onOpen(s.key); }}
                    >
                      <span className="home-card-head">
                        <span className="home-card-icon" aria-hidden="true">{s.icon}</span>
                        <span className="home-card-num" aria-hidden="true">{String(number).padStart(2, '0')}</span>
                      </span>
                      <span className="home-card-title">{st.label}</span>
                      <span className="home-card-blurb">{st.blurb}</span>
                    </a>
                  </li>
                );
              })}
            </ol>
          </section>
        );
      })}
    </div>
  );
}
