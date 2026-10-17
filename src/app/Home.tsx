import { GROUPS, SECTIONS, sectionsIn, type SectionKey } from './sections';
import { SITE_DESCRIPTION, SITE_TAGLINE, SITE_TITLE } from './site';
import './home.css';

interface HomeProps {
  onOpen: (key: SectionKey) => void;
}

/** Landing page: what the playground is, and every lesson as a card. */
export function Home({ onOpen }: HomeProps) {
  return (
    <div className="home">
      <section className="home-hero" aria-labelledby="home-title">
        <h1 id="home-title" className="home-title">{SITE_TITLE}</h1>
        <p className="home-tagline">{SITE_TAGLINE}</p>
        <p className="home-intro">{SITE_DESCRIPTION}</p>
        <div className="home-actions">
          {GROUPS.map((group, i) => (
            <a
              key={group.key}
              href={`#/${group.start}`}
              className={`btn home-cta${i === 0 ? ' btn-primary' : ''}`}
              onClick={(e) => { e.preventDefault(); onOpen(group.start); }}
            >
              {group.cta} →
            </a>
          ))}
          <button type="button" className="btn home-cta" onClick={() => onOpen(SECTIONS[Math.floor(Math.random() * SECTIONS.length)].key)}>
            🎲 Surprise me
          </button>
        </div>
        <p className="home-count">{SECTIONS.length} lessons · runs entirely in your browser · bring your own images</p>
      </section>

      {GROUPS.map(group => (
        <section key={group.key} className="home-group" aria-labelledby={`home-group-${group.key}`}>
          <header className="home-group-head">
            <h2 id={`home-group-${group.key}`} className="home-group-title">
              <span aria-hidden="true">{group.icon}</span> {group.label}
              <span className="home-group-count">{sectionsIn(group.key).length} lessons</span>
            </h2>
            <p className="home-group-blurb">{group.blurb}</p>
          </header>
          <ol className="home-grid">
            {sectionsIn(group.key).map((s, i) => (
              <li key={s.key}>
                <a
                  href={`#/${s.key}`}
                  className="home-card"
                  onClick={(e) => { e.preventDefault(); onOpen(s.key); }}
                >
                  <span className="home-card-head">
                    <span className="home-card-icon" aria-hidden="true">{s.icon}</span>
                    <span className="home-card-num" aria-hidden="true">{String(i + 1).padStart(2, '0')}</span>
                  </span>
                  <span className="home-card-title">{s.label}</span>
                  <span className="home-card-blurb">{s.blurb}</span>
                </a>
              </li>
            ))}
          </ol>
        </section>
      ))}
    </div>
  );
}
