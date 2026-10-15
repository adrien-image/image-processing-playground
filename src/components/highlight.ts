/**
 * Minimal TypeScript/TSX and Python syntax highlighter used by the code viewer.
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

const PY_KEYWORDS = new Set([
  'import', 'from', 'as', 'def', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not',
  'and', 'or', 'is', 'class', 'with', 'try', 'except', 'finally', 'raise', 'pass', 'break',
  'continue', 'lambda', 'yield', 'global', 'nonlocal', 'assert', 'del', 'async', 'await',
  'None', 'True', 'False', 'self',
]);

const PY_TYPES = new Set(['int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'bytes']);

/** String prefixes that may come right before a quote in Python (f"…", r'…', rb"…"). */
const PY_STRING_PREFIX = /^(?:[rRbBuUfF]|[rR][bBfF]|[bBfF][rR])$/;

export type Language = 'ts' | 'py';

const isIdentStart = (c: string) => /[A-Za-z_$]/.test(c);
const isIdentPart = (c: string) => /[\w$]/.test(c);
const isDigit = (c: string) => c >= '0' && c <= '9';

/** Split source code into highlighted tokens, grouped by line. */
export function highlight(code: string, language: Language = 'ts'): Token[][] {
  return toLines(language === 'py' ? tokenizePython(code) : tokenizeTs(code));
}

function tokenizeTs(code: string): Token[] {
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
  return tokens;
}

/** Python: # comments, '…' "…" and triple-quoted strings (with prefixes), @decorators. */
function tokenizePython(code: string): Token[] {
  const tokens: Token[] = [];
  const push = (type: TokenType, text: string) => {
    const last = tokens[tokens.length - 1];
    if (last && last.type === type) last.text += text;
    else tokens.push({ type, text });
  };
  const readString = (start: number): number => {
    const q = code[start];
    const triple = code.startsWith(q.repeat(3), start);
    if (triple) {
      const end = code.indexOf(q.repeat(3), start + 3);
      return end === -1 ? code.length : end + 3;
    }
    let j = start + 1;
    while (j < code.length && code[j] !== q && code[j] !== '\n') {
      if (code[j] === '\\') j++;
      j++;
    }
    return Math.min(code.length, j + 1);
  };

  let i = 0;
  const n = code.length;
  while (i < n) {
    const c = code[i];
    if (c === '#') {
      let j = code.indexOf('\n', i);
      if (j === -1) j = n;
      push('cmt', code.slice(i, j));
      i = j;
    } else if (c === '"' || c === "'") {
      const j = readString(i);
      push('str', code.slice(i, j));
      i = j;
    } else if (c === '@' && isIdentStart(code[i + 1] ?? '')) {
      let j = i + 1;
      while (j < n && (isIdentPart(code[j]) || code[j] === '.')) j++;
      push('kw', code.slice(i, j));
      i = j;
    } else if (isDigit(c) || (c === '.' && i + 1 < n && isDigit(code[i + 1]))) {
      const m = /^(?:0[xX][\da-fA-F_]+|[\d_]*\.?[\d_]+(?:[eE][+-]?\d+)?j?)/.exec(code.slice(i));
      const text = m ? m[0] : c;
      push('num', text);
      i += text.length;
    } else if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentPart(code[j])) j++;
      const word = code.slice(i, j);
      if ((code[j] === '"' || code[j] === "'") && PY_STRING_PREFIX.test(word)) {
        const end = readString(j);
        push('str', code.slice(i, end));
        i = end;
        continue;
      }
      if (PY_KEYWORDS.has(word)) push('kw', word);
      else if (PY_TYPES.has(word) || /^[A-Z][a-z]/.test(word)) push('type', word);
      else push('text', word);
      i = j;
    } else {
      push('text', c);
      i++;
    }
  }
  return tokens;
}

function toLines(tokens: Token[]): Token[][] {
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
