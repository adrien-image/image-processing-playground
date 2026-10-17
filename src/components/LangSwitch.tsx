import { setLang, useLang } from '../lib/i18n';

/** Header button switching between English and French; it shows the language it switches to. */
export function LangSwitch() {
  const lang = useLang();
  const next = lang === 'en' ? 'fr' : 'en';
  const label = lang === 'en' ? 'Lire en français' : 'Read in English';
  return (
    <button type="button" className="theme-switch lang-switch" onClick={() => setLang(next)} aria-label={label} title={label} lang={next}>
      <span aria-hidden="true">{next.toUpperCase()}</span>
    </button>
  );
}
