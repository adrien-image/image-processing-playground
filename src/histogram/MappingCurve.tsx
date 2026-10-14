/** Small SVG plot of a 256-entry gray-level mapping (input → output). */
export function MappingCurve({ mapping, size }: { mapping: Float64Array; size: number }) {
  const points = Array.from(mapping, (v, i) => `${i},${255 - v}`).join(' ');
  return (
    <svg className="mapping-curve" width={size} height={size} viewBox="-8 -8 272 272" role="img"
      aria-label="Equalisation mapping from input level to output level">
      <rect x="0" y="0" width="255" height="255" className="mapping-curve-frame" />
      <line x1="0" y1="255" x2="255" y2="0" className="mapping-curve-identity" />
      <polyline points={points} className="mapping-curve-line" />
    </svg>
  );
}
