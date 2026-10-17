import { useSyncExternalStore } from 'react';

/**
 * English / French. The choice is remembered; on a first visit it follows the
 * browser's language. Every text that differs by language is written as
 * `{ en, fr }` with the same shape (see `Texts`), and picked with `useText`.
 *
 * The source code shown by "Show code" stays in English, like the code itself.
 */
export type Lang = 'en' | 'fr';

export const LANGS: Lang[] = ['en', 'fr'];

/** The same texts in every language: `fr` must have exactly `en`'s shape. */
export interface Texts<T> {
  en: T;
  fr: T;
}

const STORAGE_KEY = 'lang';
const listeners = new Set<() => void>();

function initialLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === 'en' || v === 'fr') return v;
  } catch {
    // Storage may be unavailable.
  }
  return navigator.language?.toLowerCase().startsWith('fr') ? 'fr' : 'en';
}

let lang: Lang = initialLang();

function apply() {
  document.documentElement.lang = lang;
}

/** Apply the remembered language; call once, before the first render. */
export function initLang(): void {
  apply();
}

export function setLang(next: Lang): void {
  lang = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // Not remembering is fine.
  }
  apply();
  listeners.forEach(l => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export const useLang = (): Lang => useSyncExternalStore(subscribe, () => lang);

/** The current language's version of `texts`. */
export function useText<T>(texts: Texts<T>): T {
  return texts[useLang()];
}

/** The language outside React (e.g. for document.title in an effect that depends on useLang). */
export const currentLang = (): Lang => lang;

/** Lower case without accents, for searching: "Réseau" matches "reseau". */
export function fold(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
}
