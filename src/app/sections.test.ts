import { describe, expect, it } from 'vitest';
import { searchSections } from './search';
import { GROUPS, SECTIONS, SECTION_KEYS } from './sections';

describe('lesson list', () => {
  it('registers every key once, with French text and valid related lessons', () => {
    expect(SECTIONS.map(s => s.key).sort()).toEqual([...SECTION_KEYS].sort());
    for (const s of SECTIONS) {
      expect(s.fr.label.length).toBeGreaterThan(0);
      expect(s.fr.blurb.length).toBeGreaterThan(20);
      expect(s.related.length).toBeGreaterThan(0);
      expect(s.related).not.toContain(s.key);
      for (const r of s.related) expect(SECTION_KEYS).toContain(r);
    }
    for (const g of GROUPS) expect(SECTIONS.find(s => s.key === g.start)?.group).toBe(g.key);
  });
});

describe('searchSections', () => {
  it('matches keys, labels and blurbs in both languages, ignoring case and accents', () => {
    const keys = (q: string) => searchSections(SECTIONS, q).map(s => s.key);
    expect(keys('')).toHaveLength(SECTIONS.length);
    expect(keys('SVM')).toContain('svm');
    expect(keys('réseau récurrent')).toEqual(['rnn']);
    expect(keys('reseau recurrent')).toEqual(['rnn']);
    expect(keys('canny')).toContain('edges');
    expect(keys('zzzz')).toEqual([]);
  });
});
