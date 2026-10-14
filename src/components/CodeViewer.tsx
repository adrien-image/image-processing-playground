import { useEffect, useId, useMemo, useRef, type KeyboardEvent, type MouseEvent } from 'react';
import { highlight } from './highlight';
import './CodeViewer.css';

interface CodeViewerProps {
  code: string;
  title: string;
  onClose: () => void;
}

/** Modal dialog showing a source file with syntax highlighting. */
export function CodeViewer({ code, title, onClose }: CodeViewerProps) {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const lines = useMemo(() => highlight(code), [code]);

  // Move focus into the dialog, and give it back to the opener on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => opener?.focus();
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key !== 'Tab' || !dialogRef.current) return;
    // Keep focus inside the dialog.
    const focusables = dialogRef.current.querySelectorAll<HTMLElement>('button, [tabindex="0"]');
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const handleBackdropClick = (e: MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div className="code-viewer-overlay" role="presentation" onClick={handleBackdropClick} onKeyDown={handleKeyDown}>
      <div className="code-viewer" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="code-viewer-header">
          <span className="code-viewer-title" id={titleId}>{title}</span>
          <button ref={closeRef} type="button" className="code-viewer-close" onClick={onClose} aria-label="Close code viewer">
            ✕
          </button>
        </div>
        <pre className="code-viewer-content" role="region" tabIndex={0} aria-label={`Source code of ${title}`}>
          <code>
            {lines.map((line, i) => (
              <div key={i} className="code-viewer-line">
                <span className="code-viewer-ln" aria-hidden="true">{i + 1}</span>
                <span>
                  {line.length === 0 ? ' ' : line.map((tok, j) =>
                    tok.type === 'text' ? tok.text : <span key={j} className={`cv-${tok.type}`}>{tok.text}</span>,
                  )}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}

/** "💻 Show code" button + viewer, the pattern every section uses. */
export function ShowCodeButton({ code, title, open, onOpenChange }: {
  code: string;
  title: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <>
      <button type="button" className="btn btn-sm" onClick={() => onOpenChange(true)}>💻 Show code</button>
      {open && <CodeViewer code={code} title={title} onClose={() => onOpenChange(false)} />}
    </>
  );
}
