import { fold } from '../lib/i18n';
import type { SectionDef } from './sections';

/**
 * The sections matching a search, in menu order. Every word of the query must appear
 * in the lesson's key, label or blurb, in either language, ignoring case and accents:
 * "svm", "réseau" and "reseau" all work.
 */
export function searchSections(sections: readonly SectionDef[], query: string): SectionDef[] {
  const words = fold(query).split(/\s+/).filter(Boolean);
  if (!words.length) return [...sections];
  return sections.filter(s => {
    const haystack = fold([s.key, s.label, s.blurb, s.fr.label, s.fr.blurb].join(' '));
    return words.every(w => haystack.includes(w));
  });
}
