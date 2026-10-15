import { describe, expect, it } from 'vitest';
import { highlight, type Token } from './highlight';

const types = (line: Token[]) => line.filter(t => t.text.trim()).map(t => `${t.type}:${t.text.trim()}`);

describe('highlight', () => {
  it('marks every line of a multi-line JSDoc block as a comment', () => {
    const lines = highlight('/**\n * Returns the new value\n */\nconst x = 1;');
    expect(lines).toHaveLength(4);
    for (const l of lines.slice(0, 3)) expect(l.every(t => t.type === 'cmt')).toBe(true);
    expect(types(lines[3])).toEqual(['kw:const', 'text:x =', 'num:1', 'text:;']);
  });

  it('does not treat // inside a string as a comment', () => {
    const [line] = highlight("const url = 'https://example.com'; // real comment");
    expect(types(line)).toEqual([
      'kw:const', 'text:url =', "str:'https://example.com'", 'text:;', 'cmt:// real comment',
    ]);
  });

  it('keeps digits inside identifiers as text and handles escapes', () => {
    const [line] = highlight('let a2b = "q\\"x" + 3.5e2;');
    expect(types(line)).toEqual(['kw:let', 'text:a2b =', 'str:"q\\"x"', 'text:+', 'num:3.5e2', 'text:;']);
  });

  it('highlights types and PascalCase names', () => {
    const [line] = highlight('function f(a: number): Float64Array {}');
    expect(types(line)).toContain('type:number');
    expect(types(line)).toContain('type:Float64Array');
  });

  it('preserves the exact source text', () => {
    const src = 'a /* x\ny */ `t\n${b}` // c\n\nend';
    const rebuilt = highlight(src).map(l => l.map(t => t.text).join('')).join('\n');
    expect(rebuilt).toBe(src);
  });

  it('highlights Python: # comments, keywords, triple-quoted and prefixed strings, decorators', () => {
    const src = [
      '@torch.no_grad()',
      'def f(x):  # a comment',
      '    """Doc',
      '    string"""',
      '    return f"{x}" if x is None else 1.5e-3',
    ].join('\n');
    const lines = highlight(src, 'py');
    expect(types(lines[0])).toEqual(['kw:@torch.no_grad', 'text:()']);
    expect(types(lines[1])).toEqual(['kw:def', 'text:f(x):', 'cmt:# a comment']);
    expect(types(lines[2])).toEqual(['str:"""Doc']);
    expect(types(lines[3])).toEqual(['str:string"""']);
    expect(types(lines[4])).toEqual(['kw:return', 'str:f"{x}"', 'kw:if', 'text:x', 'kw:is', 'kw:None', 'kw:else', 'num:1.5e-3']);
    expect(lines.map(l => l.map(t => t.text).join('')).join('\n')).toBe(src);
  });

  it('does not treat # inside a Python string as a comment', () => {
    const [line] = highlight("x = '#not a comment'  # real", 'py');
    expect(types(line)).toEqual(['text:x =', "str:'#not a comment'", 'cmt:# real']);
  });
});
