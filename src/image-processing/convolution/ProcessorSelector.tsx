import { useId } from 'react';
import { getAllProcessors, type ParamDefinition, type ParamValue, type ProcessorConfig } from '../processors';

interface ProcessorSelectorProps {
  activeProcessor: string | null;
  config: ProcessorConfig;
  onSelect: (processorId: string) => void;
  onConfigChange: (config: ProcessorConfig) => void;
  disabled?: boolean;
}

/**
 * Lists every registered processor and renders controls from the active one's
 * `params` schema.
 */
export function ProcessorSelector({ activeProcessor, config, onSelect, onConfigChange, disabled }: ProcessorSelectorProps) {
  const selectId = useId();
  const processors = getAllProcessors();
  const activeProc = processors.find(p => p.id === activeProcessor);

  return (
    <div className="processor-selector">
      <h3><label htmlFor={selectId}>Processor</label></h3>
      <select
        id={selectId}
        className="select"
        onChange={(e) => { if (e.target.value) onSelect(e.target.value); }}
        value={activeProcessor ?? ''}
        disabled={disabled}
      >
        <option value="">— Select a processor —</option>
        {processors.map(p => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>

      {activeProc && (
        <div className="processor-info">
          <p className="processor-desc">{activeProc.description}</p>
          {Object.keys(activeProc.params).length > 0 && (
            <div className="processor-params">
              {Object.entries(activeProc.params).map(([key, param]) => (
                <ParamControl
                  key={key}
                  def={param}
                  value={config[key] ?? param.default}
                  onChange={(v) => onConfigChange({ ...config, [key]: v })}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ParamControl({ def, value, onChange }: { def: ParamDefinition; value: ParamValue; onChange: (v: ParamValue) => void }) {
  if (def.type === 'boolean') {
    return (
      <label className="param-row">
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        {def.label}
      </label>
    );
  }

  if (def.type === 'select' && def.options) {
    return (
      <label className="param-row">
        <span>{def.label}</span>
        <select className="select" value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {def.options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
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
