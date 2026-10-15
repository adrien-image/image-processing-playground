import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import rfCode from './receptiveField.ts?raw';
import pythonCode from './receptive_field.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { hotColor } from '../../image-processing/hough/hough';
import { dependencyRange, fieldRange, layerGeometry, pathCounts, PRESETS, type LayerKind, type LayerSpec } from './receptiveField';
import '../shared/dl-shared.css';
import './receptive-field.css';

const PANEL = 200;
const MAX_LAYERS = 8;

interface Selection { layer: number; x: number; y: number }

export function ReceptiveFieldSection() {
  const [presetKey, setPresetKey] = useState<string>('lenet');
  const [inputSize, setInputSize] = useState(PRESETS.lenet.inputSize);
  const [layers, setLayers] = useState<LayerSpec[]>(PRESETS.lenet.layers);
  const [showPaths, setShowPaths] = useState(true);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [showCode, setShowCode] = useState(false);

  const geometry = useMemo(() => layerGeometry(inputSize, layers), [inputSize, layers]);
  const invalidAt = geometry.findIndex(g => g.size < 1);
  const valid = invalidAt === -1;
  const last = geometry.length - 1;

  // Default selection: the centre unit of the last layer.
  const sel: Selection | null = valid
    ? selection && selection.layer <= last && selection.x < geometry[selection.layer].size && selection.y < geometry[selection.layer].size
      ? selection
      : { layer: last, x: geometry[last].size >> 1, y: geometry[last].size >> 1 }
    : null;

  const selLayer = sel?.layer ?? -1, selX = sel?.x ?? 0, selY = sel?.y ?? 0;
  const counts = useMemo(() => {
    if (selLayer < 0) return null;
    return { cx: pathCounts(layers, geometry, selLayer, selX), cy: pathCounts(layers, geometry, selLayer, selY) };
  }, [layers, geometry, selLayer, selX, selY]);

  const loadPreset = (key: string) => {
    setPresetKey(key);
    setInputSize(PRESETS[key].inputSize);
    setLayers(PRESETS[key].layers);
    setSelection(null);
  };
  const edit = (i: number, patch: Partial<LayerSpec>) => {
    setPresetKey('custom');
    setLayers(ls => ls.map((l, j) => (j === i ? { ...l, ...patch } : l)));
  };

  const selected = sel ? geometry[sel.layer] : null;
  const [fx0, fx1] = sel && selected ? fieldRange(selected, sel.x) : [0, 0];
  const [fy0, fy1] = sel && selected ? fieldRange(selected, sel.y) : [0, 0];

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Network:</span>
        {Object.entries(PRESETS).map(([key, p]) => (
          <button key={key} type="button" aria-pressed={presetKey === key}
            className={`btn btn-sm${presetKey === key ? ' btn-primary' : ''}`} onClick={() => loadPreset(key)}>
            {p.label}
          </button>
        ))}
        <span className="ml-auto">
          <ShowCodeButton code={rfCode} title="receptive-field/receptiveField.ts" python={{ code: pythonCode, title: 'receptive-field/receptive_field.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls rf-editor">
        <label className="dl-slider">
          Input size <strong>{inputSize}×{inputSize}</strong>
          <input type="range" min={8} max={64} value={inputSize} onChange={e => { setInputSize(Number(e.target.value)); setPresetKey('custom'); }} />
        </label>
        <label className="dl-slider">
          <input type="checkbox" checked={showPaths} onChange={e => setShowPaths(e.target.checked)} /> Shade by number of paths
        </label>
        <table className="rf-table">
          <thead>
            <tr><th>Layer</th><th>Type</th><th>Kernel</th><th>Stride</th><th>Padding</th><th>Dilation</th><th>Output</th><th>Field</th><th>Jump</th><th><span className="visually-hidden">Remove</span></th></tr>
          </thead>
          <tbody>
            {layers.map((l, i) => {
              const g = geometry[i + 1];
              return (
                <tr key={i} className={invalidAt === i + 1 ? 'rf-invalid' : undefined}>
                  <td className="mono">{i + 1}</td>
                  <td>
                    <select className="select rf-input" value={l.kind} aria-label={`Layer ${i + 1} type`}
                      onChange={e => edit(i, { kind: e.target.value as LayerKind })}>
                      <option value="conv">conv</option>
                      <option value="pool">pool</option>
                    </select>
                  </td>
                  <NumberCell value={l.kernel} min={1} max={11} label={`Layer ${i + 1} kernel`} onChange={v => edit(i, { kernel: v })} />
                  <NumberCell value={l.stride} min={1} max={4} label={`Layer ${i + 1} stride`} onChange={v => edit(i, { stride: v })} />
                  <NumberCell value={l.padding} min={0} max={8} label={`Layer ${i + 1} padding`} onChange={v => edit(i, { padding: v })} />
                  <NumberCell value={l.dilation} min={1} max={8} label={`Layer ${i + 1} dilation`} onChange={v => edit(i, { dilation: v })} />
                  <td className="mono">{g.size >= 1 ? `${g.size}×${g.size}` : '—'}</td>
                  <td className="mono">{g.field}</td>
                  <td className="mono">{g.jump}</td>
                  <td>
                    <button type="button" className="btn btn-sm" aria-label={`Remove layer ${i + 1}`}
                      onClick={() => { setPresetKey('custom'); setLayers(ls => ls.filter((_, j) => j !== i)); }}>✕</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="rf-add">
          <button type="button" className="btn btn-sm" disabled={layers.length >= MAX_LAYERS}
            onClick={() => { setPresetKey('custom'); setLayers(ls => [...ls, { kind: 'conv', kernel: 3, stride: 1, padding: 1, dilation: 1 }]); }}>
            + conv 3×3
          </button>
          <button type="button" className="btn btn-sm" disabled={layers.length >= MAX_LAYERS}
            onClick={() => { setPresetKey('custom'); setLayers(ls => [...ls, { kind: 'pool', kernel: 2, stride: 2, padding: 0, dilation: 1 }]); }}>
            + pool 2×2
          </button>
        </div>
      </div>

      {!valid && (
        <div className="error-banner" role="alert">
          ⚠ Layer {invalidAt} would have no output: its kernel is wider than its (padded) input. Add padding, shrink the kernel or enlarge the input.
        </div>
      )}

      {sel && selected && counts && (
        <>
          <p className="rf-summary">
            Selected: <strong>{sel.layer === 0 ? 'input pixel' : `layer ${sel.layer} unit`} ({sel.x}, {sel.y})</strong> · its receptive field is
            {' '}<strong>{selected.field}×{selected.field}</strong> input pixels, x {fmt(fx0)}…{fmt(fx1)}, y {fmt(fy0)}…{fmt(fy1)}
            {(fx0 < 0 || fy0 < 0 || fx1 >= inputSize || fy1 >= inputSize) && ' (part of it lies in the padding, outside the image)'}.
            Click any unit to inspect it.
          </p>
          <div className="rf-layers">
            {geometry.map((g, l) => {
              const rx = l <= sel.layer ? dependencyRange(layers, geometry, sel.layer, [sel.x, sel.x], l) : null;
              const ry = l <= sel.layer ? dependencyRange(layers, geometry, sel.layer, [sel.y, sel.y], l) : null;
              return (
                <div key={l} className="demo-panel rf-panel">
                  <h3>{l === 0 ? 'Input' : `${l}: ${layers[l - 1].kind} ${layers[l - 1].kernel}×${layers[l - 1].kernel}`}</h3>
                  <LayerCanvas
                    size={g.size}
                    rangeX={rx}
                    rangeY={ry}
                    selected={l === sel.layer ? [sel.x, sel.y] : null}
                    heat={l === 0 && showPaths ? counts : null}
                    onPick={(x, y) => setSelection({ layer: l, x, y })}
                    label={`${l === 0 ? 'Input' : `Layer ${l}`}, ${g.size} by ${g.size} units`}
                  />
                  <p className="caption">{g.size}×{g.size} · field {g.field} · jump {g.jump}</p>
                </div>
              );
            })}
          </div>
        </>
      )}

      <div className="demo-explanation">
        <h4>How far can a unit see?</h4>
        <p>A unit in the first convolution sees only its k×k window. A unit in the next layer combines several of those
        windows, so it sees further: each layer widens the field by <code>(k − 1) × jump</code>, where the jump is the product
        of all earlier strides. That is why strided layers and pooling make later layers grow their field so fast, and why
        <strong> dilated</strong> convolutions (gaps between kernel taps) can cover a large area with few layers.</p>
        <p>The shading on the input counts how many paths through the network connect each pixel to the selected unit.
        Pixels at the edge of the field are reached by a single path, the centre by many, so in practice a unit is mostly
        sensitive to the middle of its theoretical field: the <strong>effective receptive field</strong> is roughly Gaussian.</p>
        <p>The table shows LeNet's last pooling layer seeing 16×16 pixels of the 28×28 digit. Compare it with the occlusion
        map in the Classification lesson, which measures what the network actually uses.</p>
      </div>
    </div>
  );
}

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

function NumberCell({ value, min, max, label, onChange }: { value: number; min: number; max: number; label: string; onChange: (v: number) => void }) {
  return (
    <td>
      <input className="rf-input rf-number" type="number" min={min} max={max} value={value} aria-label={label}
        onChange={e => { const v = Number(e.target.value); if (Number.isInteger(v) && v >= min && v <= max) onChange(v); }} />
    </td>
  );
}

interface LayerCanvasProps {
  size: number;
  rangeX: [number, number] | null;
  rangeY: [number, number] | null;
  selected: [number, number] | null;
  heat: { cx: Float64Array; cy: Float64Array } | null;
  onPick: (x: number, y: number) => void;
  label: string;
}

/** One layer's units as a grid: the dependency range shaded, the selected unit outlined. */
function LayerCanvas({ size, rangeX, rangeY, selected, heat, onPick, label }: LayerCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = Math.max(2, Math.floor(PANEL / size));
  const px = cell * size;

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#141824';
    ctx.fillRect(0, 0, px, px);
    if (rangeX && rangeY) {
      let max = 0;
      if (heat) for (let y = rangeY[0]; y <= rangeY[1]; y++) for (let x = rangeX[0]; x <= rangeX[1]; x++) max = Math.max(max, heat.cx[x] * heat.cy[y]);
      for (let y = rangeY[0]; y <= rangeY[1]; y++) {
        for (let x = rangeX[0]; x <= rangeX[1]; x++) {
          if (heat) {
            const [r, g, b] = hotColor(0.15 + 0.85 * ((heat.cx[x] * heat.cy[y]) / (max || 1)));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
          } else {
            ctx.fillStyle = 'rgba(0, 240, 255, 0.45)';
          }
          ctx.fillRect(x * cell, y * cell, cell, cell);
        }
      }
    }
    if (cell >= 5) {
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= size; i++) {
        ctx.moveTo(i * cell + 0.5, 0); ctx.lineTo(i * cell + 0.5, px);
        ctx.moveTo(0, i * cell + 0.5); ctx.lineTo(px, i * cell + 0.5);
      }
      ctx.stroke();
    }
    if (selected) {
      ctx.strokeStyle = '#ffdd44';
      ctx.lineWidth = 2;
      ctx.strokeRect(selected[0] * cell + 1, selected[1] * cell + 1, cell - 2 > 0 ? cell - 2 : cell, cell - 2 > 0 ? cell - 2 : cell);
    }
  }, [size, cell, px, rangeX, rangeY, selected, heat]);

  const click = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * size);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    onPick(Math.min(size - 1, Math.max(0, x)), Math.min(size - 1, Math.max(0, y)));
  };

  return <canvas ref={ref} width={px} height={px} className="rf-canvas" onClick={click} role="img" aria-label={label} />;
}
