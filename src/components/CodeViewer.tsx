import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { highlight, type Language } from './highlight';
import './CodeViewer.css';

export interface SourceFile {
  title: string;
  code: string;
  language: Language;
}

const LANGUAGE_LABEL: Record<Language, string> = { ts: 'TypeScript', py: 'Python · PyTorch' };
const PREFERENCE_KEY = 'code-viewer-language';

/** The language the viewer last showed, remembered per browser (storage may be unavailable). */
function preferredLanguage(): Language | null {
  try {
    const v = localStorage.getItem(PREFERENCE_KEY);
    return v === 'ts' || v === 'py' ? v : null;
  } catch {
    return null;
  }
}

function rememberLanguage(language: Language) {
  try {
    localStorage.setItem(PREFERENCE_KEY, language);
  } catch {
    // Not remembering is fine.
  }
}

interface CodeViewerProps {
  /** One file per language; with more than one, the viewer shows a tab per language. */
  files: SourceFile[];
  onClose: () => void;
}

/** Modal dialog showing a source file with syntax highlighting, with a tab per language. */
export function CodeViewer({ files, onClose }: CodeViewerProps) {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const [index, setIndex] = useState(() => Math.max(0, files.findIndex(f => f.language === preferredLanguage())));
  const file = files[Math.min(index, files.length - 1)];
  const { title } = file;
  const lines = useMemo(() => highlight(file.code, file.language), [file]);

  const select = (i: number) => {
    setIndex(i);
    rememberLanguage(files[i].language);
  };

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
          {files.length > 1 && (
            <div className="code-viewer-tabs" role="tablist" aria-label="Language">
              {files.map((f, i) => (
                <button key={f.language} type="button" role="tab" aria-selected={f === file}
                  className={`code-viewer-tab${f === file ? ' code-viewer-tab-active' : ''}`} onClick={() => select(i)}>
                  {LANGUAGE_LABEL[f.language]}
                </button>
              ))}
            </div>
          )}
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
export function ShowCodeButton({ code, title, python, open, onOpenChange }: {
  code: string;
  title: string;
  /** The same algorithm in Python / PyTorch, shown in a second tab. */
  python?: { code: string; title: string };
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const files: SourceFile[] = [{ code, title, language: 'ts' }];
  if (python) files.push({ ...python, language: 'py' });
  return (
    <>
      <button type="button" className="btn btn-sm" onClick={() => onOpenChange(true)}>💻 Show code</button>
      {open && <CodeViewer files={files} onClose={() => onOpenChange(false)} />}
    </>
  );
}
