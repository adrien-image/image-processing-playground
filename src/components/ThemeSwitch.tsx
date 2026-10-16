import { setThemeChoice, useThemeChoice, type ThemeChoice } from '../lib/theme';

const NEXT: Record<ThemeChoice, ThemeChoice> = { system: 'light', light: 'dark', dark: 'system' };
const ICON: Record<ThemeChoice, string> = { system: '🖥', light: '☀️', dark: '🌙' };
const LABEL: Record<ThemeChoice, string> = { system: 'System', light: 'Light', dark: 'Dark' };

/** Header button cycling System → Light → Dark. */
export function ThemeSwitch() {
  const choice = useThemeChoice();
  const next = NEXT[choice];
  return (
    <button
      type="button"
      className="theme-switch"
      onClick={() => setThemeChoice(next)}
      aria-label={`Theme: ${LABEL[choice]}. Switch to ${LABEL[next].toLowerCase()}`}
      title={`Theme: ${LABEL[choice]} (click for ${LABEL[next].toLowerCase()})`}
    >
      <span aria-hidden="true">{ICON[choice]}</span>
      <span className="theme-switch-label">{LABEL[choice]}</span>
    </button>
  );
}
