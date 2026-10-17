import { useText } from '../lib/i18n';
import { setThemeChoice, useThemeChoice, type ThemeChoice } from '../lib/theme';

const NEXT: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' };
const ICON: Record<ThemeChoice, string> = { system: '🖥', light: '☀️', dark: '🌙' };
const TEXT = {
  en: { names: { system: 'System', light: 'Light', dark: 'Dark' }, theme: 'Theme', switchTo: 'Switch to', clickFor: 'click for' },
  fr: { names: { system: 'Système', light: 'Clair', dark: 'Sombre' }, theme: 'Thème', switchTo: 'Passer en', clickFor: 'cliquer pour' },
};

/** Header button cycling System → Light → Dark. */
export function ThemeSwitch() {
  const choice = useThemeChoice();
  const t = useText(TEXT);
  const next = NEXT[choice];
  return (
    <button
      type="button"
      className="theme-switch"
      onClick={() => setThemeChoice(next)}
      aria-label={`${t.theme}: ${t.names[choice]}. ${t.switchTo} ${t.names[next].toLowerCase()}`}
      title={`${t.theme}: ${t.names[choice]} (${t.clickFor} ${t.names[next].toLowerCase()})`}
    >
      <span aria-hidden="true">{ICON[choice]}</span>
      <span className="theme-switch-label">{t.names[choice]}</span>
    </button>
  );
}
