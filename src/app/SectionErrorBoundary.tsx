import { Component, type ReactNode } from 'react';

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
    return (
      <div className="error-banner section-error" role="alert">
        <p>⚠ This section failed to load: {this.state.error.message}</p>
        <button type="button" className="btn btn-sm" onClick={() => window.location.reload()}>Reload page</button>
      </div>
    );
  }
}
