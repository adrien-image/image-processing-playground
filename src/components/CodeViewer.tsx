import { useEffect, useCallback, useMemo } from 'react';

interface CodeViewerProps {
  code: string;
  title: string;
  onClose: () => void;
}

const KEYWORDS = new Set([
  'import', 'export', 'default', 'from', 'as',
  'const', 'let', 'var', 'function', 'return', 'if', 'else',
  'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
  'class', 'interface', 'type', 'extends', 'implements', 'enum',
  'new', 'this', 'super', 'throw', 'try', 'catch', 'finally',
  'async', 'await', 'yield', 'of', 'in', 'instanceof', 'typeof',
  'true', 'false', 'null', 'undefined', 'void', 'never',
  'public', 'private', 'protected', 'readonly', 'static', 'abstract',
]);

const TYPES = new Set([
  'string', 'number', 'boolean', 'void', 'never', 'any', 'unknown',
  'Float64Array', 'Uint8Array', 'Uint32Array', 'Uint8ClampedArray', 'Float32Array',
  'Int8Array', 'Int16Array', 'Int32Array', 'ImageData', 'HTMLImageElement',
  'HTMLCanvasElement', 'CanvasRenderingContext2D', 'Complex', 'Keypoint',
  'Match', 'SimTransformResult', 'KMeansResult', 'DetectedLine', 'DetectedCircle',
  'Promise', 'Map', 'Set', 'Array', 'Record', 'Partial', 'Required', 'Pick', 'Omit',
  'ReturnType', 'Awaited', 'Parameters',
]);

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function highlightLine(line: string): string {
  // HTML-escape first, then apply syntax highlights on safe text
  let result = escapeHtml(line);

  // Strings (match after escaping: &quot;...&quot;  &#39;...&#39;  `...`)
  result = result.replace(/&quot;(?:(?!&quot;).)*&quot;/g, '<span class="cv-str">$&</span>');
  result = result.replace(/&#39;(?:(?!&#39;).)*&#39;/g, '<span class="cv-str">$&</span>');
  result = result.replace(/`(?:(?!`).)*`/g, '<span class="cv-str">$&</span>');

  // Block comments /* ... */
  result = result.replace(/\/\*[\s\S]*?\*\//g, '<span class="cv-cmt">$&</span>');
  // Line comments //
  result = result.replace(/\/\/.*/g, '<span class="cv-cmt">$&</span>');

  // Numbers
  result = result.replace(/\b(\d+\.?\d*)\b/g, (m, n, idx) => {
    // Don't highlight if already inside a span (string or comment)
    const before = result.substring(0, idx);
    const openTags = (before.match(/<span/g) || []).length;
    const closeTags = (before.match(/<\/span>/g) || []).length;
    if (openTags > closeTags) return m;
    return `<span class="cv-num">${n}</span>`;
  });

  // Keywords and types
  result = result.replace(/\b([a-zA-Z_]\w*)\b/g, (m, word, idx) => {
    const before = result.substring(0, idx);
    const openTags = (before.match(/<span/g) || []).length;
    const closeTags = (before.match(/<\/span>/g) || []).length;
    if (openTags > closeTags) return m;
    if (KEYWORDS.has(word)) return `<span class="cv-kw">${word}</span>`;
    if (TYPES.has(word)) return `<span class="cv-type">${word}</span>`;
    // PascalCase identifiers (likely types/interfaces/components)
    if (/^[A-Z][a-z]/.test(word)) return `<span class="cv-type">${word}</span>`;
    return m;
  });

  return result;
}

export function CodeViewer({ code, title, onClose }: CodeViewerProps) {
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [handleKey]);

  const highlighted = useMemo(() =>
    code.split('\n').map(highlightLine),
  [code]);

  return (
    <div className="code-viewer-overlay" onClick={onClose}>
      <div className="code-viewer" onClick={e => e.stopPropagation()}>
        <div className="code-viewer-header">
          <span className="code-viewer-title">{title}</span>
          <button className="code-viewer-close" onClick={onClose}>✕</button>
        </div>
        <pre className="code-viewer-content">
          <code>
            {highlighted.map((line, i) => (
              <div key={i} className="code-viewer-line">
                <span className="code-viewer-ln">{i + 1}</span>
                <span dangerouslySetInnerHTML={{ __html: line || ' ' }} />
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
