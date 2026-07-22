import { useCallback } from 'react';
import { getAllProcessors } from '../processors';
import type { ParamDefinition } from '../processors';

interface ProcessorSelectorProps {
  onSelect: (processorId: string, config: Record<string, number | boolean | string>) => void;
  activeProcessor: string | null;
  disabled?: boolean;
  config: Record<string, number | boolean | string>;
  onConfigChange: (config: Record<string, number | boolean | string>) => void;
}

export function ProcessorSelector({
  onSelect,
  activeProcessor,
  disabled,
  config,
  onConfigChange,
}: ProcessorSelectorProps) {
  const processors = getAllProcessors();

  const handleProcessorChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    if (!id) return;

    // Build default config from params
    const proc = processors.find(p => p.id === id);
    if (proc) {
      const defaults: Record<string, number | boolean | string> = {};
      for (const [key, def] of Object.entries(proc.params)) {
        defaults[key] = def.default;
      }
      onConfigChange(defaults);
      onSelect(id, defaults);
    }
  }, [processors, onSelect, onConfigChange]);

  const activeProc = processors.find(p => p.id === activeProcessor);

  const updateParam = useCallback((key: string, value: number | boolean | string) => {
    const newConfig = { ...config, [key]: value };
    onConfigChange(newConfig);
    // Re-apply with new config
    if (activeProcessor) {
      onSelect(activeProcessor, newConfig);
    }
  }, [config, onConfigChange, onSelect, activeProcessor]);

  return (
    <div className="processor-selector">
      <h3>Processor</h3>
      <select
        className="select"
        onChange={handleProcessorChange}
        value={activeProcessor ?? ''}
        disabled={disabled}
      >
        <option value="">— Select a processor —</option>
        {processors.map(p => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>

      {activeProc && (
        <div className="processor-info">
          <p className="processor-desc">{activeProc.description}</p>
          <div className="processor-params">
            {Object.entries(activeProc.params).map(([key, param]) => (
              <ParamControl
                key={key}
                def={param}
                value={config[key] ?? param.default}
                onChange={(v) => updateParam(key, v)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ParamControl({
  def,
  value,
  onChange,
}: {
  def: ParamDefinition;
  value: number | boolean | string;
  onChange: (v: number | boolean | string) => void;
}) {
  if (def.type === 'boolean') {
    return (
      <label className="param-row">
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
        />
        {def.label}
      </label>
    );
  }

  if (def.type === 'select' && def.options) {
    return (
      <label className="param-row">
        <span>{def.label}</span>
        <select
          className="select"
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
        >
          {def.options.map(o => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </label>
    );
  }

  if (def.type === 'number') {
    return (
      <label className="param-row">
        <span>{def.label}: <strong>{value}</strong></span>
        <input
          type="range"
          min={def.min ?? 0}
          max={def.max ?? 255}
          step={def.step ?? 1}
          value={Number(value)}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </label>
    );
  }

  return null;
}
