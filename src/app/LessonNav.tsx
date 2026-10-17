import type { ReactNode } from 'react';
import { useLang, useText } from '../lib/i18n';
import { SECTIONS, sectionText, type SectionKey } from './sections';

const TEXT = {
  en: { nav: 'More lessons', previous: 'Previous', next: 'Next', related: 'Related lessons' },
  fr: { nav: 'Autres leçons', previous: 'Précédente', next: 'Suivante', related: 'Leçons liées' },
};

/** Under a lesson: the previous and next lessons in menu order, and the related ones. */
export function LessonNav({ current, onOpen }: { current: SectionKey; onOpen: (key: SectionKey) => void }) {
  const lang = useLang();
  const t = useText(TEXT);
  const index = SECTIONS.findIndex(s => s.key === current);
  const previous = SECTIONS[index - 1], next = SECTIONS[index + 1];
  const related = SECTIONS[index].related.map(k => SECTIONS.find(s => s.key === k)).filter(s => s !== undefined);

  const link = (key: SectionKey, children: ReactNode, className: string) => (
    <a href={`#/${key}`} className={className} onClick={e => { e.preventDefault(); onOpen(key); }}>{children}</a>
  );

  return (
    <nav className="lesson-nav" aria-label={t.nav}>
      <div className="lesson-nav-steps">
        {previous ? link(previous.key, (
          <>
            <span className="lesson-nav-dir">← {t.previous}</span>
            <span className="lesson-nav-title"><span aria-hidden="true">{previous.icon}</span> {sectionText(previous, lang).label}</span>
          </>
        ), 'lesson-nav-step') : <span />}
        {next && link(next.key, (
          <>
            <span className="lesson-nav-dir">{t.next} →</span>
            <span className="lesson-nav-title"><span aria-hidden="true">{next.icon}</span> {sectionText(next, lang).label}</span>
          </>
        ), 'lesson-nav-step lesson-nav-next')}
      </div>
      {related.length > 0 && (
        <div className="lesson-nav-related">
          <span className="lesson-nav-dir">{t.related}</span>
          <ul>
            {related.map(s => (
              <li key={s.key}>{link(s.key, <><span aria-hidden="true">{s.icon}</span> {sectionText(s, lang).label}</>, 'btn btn-sm lesson-nav-chip')}</li>
            ))}
          </ul>
        </div>
      )}
    </nav>
  );
}
