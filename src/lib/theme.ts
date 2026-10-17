import { useSyncExternalStore } from 'react';

/**
 * Light / dark theme. 'system' follows the operating system's setting through CSS
 * alone (`color-scheme: light dark`); an explicit choice sets data-theme on <html>,
 * which pins `color-scheme` (see the tokens at the top of styles/base.css).
 */
export type ThemeChoice = 'system' | 'light' | 'dark';
export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';
const listeners = new Set<() => void>();
const prefersLight = window.matchMedia('(prefers-color-scheme: light)');

/** The remembered choice (storage may be unavailable). */
function storedChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

let choice: ThemeChoice = storedChoice();

function apply() {
  if (choice === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
}

const notify = () => listeners.forEach(l => l());

/** Apply the remembered choice; call once, before the first render. */
export function initTheme(): void {
  apply();
  prefersLight.addEventListener('change', notify);
}

export function setThemeChoice(next: ThemeChoice): void {
  choice = next;
  try {
    if (next === 'system') localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // Not remembering is fine.
  }
  apply();
  notify();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const resolved = (): Theme => (choice === 'system' ? (prefersLight.matches ? 'light' : 'dark') : choice);

export const useThemeChoice = (): ThemeChoice => useSyncExternalStore(subscribe, () => choice);

/** The theme in effect. Canvas drawings depend on it so they redraw when it changes. */
export const useTheme = (): Theme => useSyncExternalStore(subscribe, resolved);

let probe: HTMLSpanElement | null = null;

/**
 * A CSS colour (e.g. 'var(--text-muted)') resolved to a concrete colour, for drawing
 * on a canvas, which cannot read CSS variables. Call it while drawing, so it reflects
 * the current theme.
 */
export function cssColor(value: string): string {
  if (!probe) {
    probe = document.createElement('span');
    probe.hidden = true;
    document.body.append(probe);
  }
  probe.style.color = value;
  return getComputedStyle(probe).color;
}
