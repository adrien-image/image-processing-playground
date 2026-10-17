import { Component, type ReactNode } from 'react';
import { useText } from '../lib/i18n';

const TEXT = {
  en: { failed: 'This section failed to load:', reload: 'Reload page' },
  fr: { failed: 'Cette leçon n\'a pas pu se charger :', reload: 'Recharger la page' },
};

function SectionError({ error }: { error: Error }) {
  const t = useText(TEXT);
  return (
    <div className="error-banner section-error" role="alert">
      <p>⚠ {t.failed} {error.message}</p>
      <button type="button" className="btn btn-sm" onClick={() => window.location.reload()}>{t.reload}</button>
    </div>
  );
}

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Catches render errors and failed chunk loads so one broken section can't blank the app. */
export class SectionErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <SectionError error={this.state.error} />;
  }
}
