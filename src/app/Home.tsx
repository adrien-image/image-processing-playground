import { SECTIONS, type SectionKey } from './sections';
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
          <a
            href="#/convolution"
            className="btn btn-primary home-cta"
            onClick={(e) => { e.preventDefault(); onOpen('convolution'); }}
          >
            Start with convolution →
          </a>
          <span className="home-count">{SECTIONS.length} lessons · runs entirely in your browser · bring your own images</span>
        </div>
      </section>

      <nav aria-label="Lessons">
        <ol className="home-grid">
          {SECTIONS.map((s, i) => (
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
      </nav>
    </div>
  );
}
