/**
 * Minimal TypeScript/TSX syntax highlighter used by the code viewer.
 *
 * It is a single-pass tokenizer (not regex replacement over HTML), so multi-line
 * comments, template literals, and `//` inside strings are handled correctly, and
 * the output is plain data rendered by React — no `innerHTML`.
 */

export type TokenType = 'kw' | 'type' | 'str' | 'cmt' | 'num' | 'text';

export interface Token {
  type: TokenType;
  text: string;
}

const KEYWORDS = new Set([
  'import', 'export', 'default', 'from', 'as',
  'const', 'let', 'var', 'function', 'return', 'if', 'else',
  'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
  'class', 'interface', 'type', 'extends', 'implements', 'enum',
  'new', 'this', 'super', 'throw', 'try', 'catch', 'finally',
  'async', 'await', 'yield', 'of', 'in', 'instanceof', 'typeof', 'keyof',
  'true', 'false', 'null', 'undefined', 'void', 'never',
  'public', 'private', 'protected', 'readonly', 'static', 'abstract',
]);

const TYPES = new Set([
  'string', 'number', 'boolean', 'any', 'unknown', 'object',
]);

const isIdentStart = (c: string) => /[A-Za-z_$]/.test(c);
const isIdentPart = (c: string) => /[\w$]/.test(c);
const isDigit = (c: string) => c >= '0' && c <= '9';

/** Split source code into highlighted tokens, grouped by line. */
export function highlight(code: string): Token[][] {
  const tokens: Token[] = [];
  const push = (type: TokenType, text: string) => {
    const last = tokens[tokens.length - 1];
    if (last && last.type === type) last.text += text;
    else tokens.push({ type, text });
  };

  let i = 0;
  const n = code.length;
  while (i < n) {
    const c = code[i];
    const next = code[i + 1];

    if (c === '/' && next === '/') {
      let j = code.indexOf('\n', i);
      if (j === -1) j = n;
      push('cmt', code.slice(i, j));
      i = j;
    } else if (c === '/' && next === '*') {
      let j = code.indexOf('*/', i + 2);
      j = j === -1 ? n : j + 2;
      push('cmt', code.slice(i, j));
      i = j;
    } else if (c === '"' || c === "'" || c === '`') {
      let j = i + 1;
      while (j < n && code[j] !== c) {
        if (code[j] === '\\') j++;
        // Plain quotes cannot span lines; stop at the newline if unterminated.
        if (c !== '`' && code[j] === '\n') break;
        j++;
      }
      j = Math.min(n, j + 1);
      push('str', code.slice(i, j));
      i = j;
    } else if (isDigit(c) || (c === '.' && next !== undefined && isDigit(next))) {
      const m = /^(?:0[xX][\da-fA-F_]+|[\d_]*\.?[\d_]+(?:[eE][+-]?\d+)?)n?/.exec(code.slice(i));
      const text = m ? m[0] : c;
      push('num', text);
      i += text.length;
    } else if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentPart(code[j])) j++;
      const word = code.slice(i, j);
      if (KEYWORDS.has(word)) push('kw', word);
      else if (TYPES.has(word) || /^[A-Z][a-z]/.test(word)) push('type', word);
      else push('text', word);
      i = j;
    } else {
      push('text', c);
      i++;
    }
  }

  // Split tokens into lines (comments and template strings may span several).
  const lines: Token[][] = [[]];
  for (const tok of tokens) {
    const parts = tok.text.split('\n');
    parts.forEach((part, k) => {
      if (k > 0) lines.push([]);
      if (part) lines[lines.length - 1].push({ type: tok.type, text: part });
    });
  }
  return lines;
}
