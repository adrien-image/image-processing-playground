import { useState, type ComponentType } from 'react';
import { ResamplingTab } from './ResamplingTab';
import { PyramidTab } from './PyramidTab';
import { BlendingTab } from './BlendingTab';
import './pyramids.css';

type TabKey = 'resampling' | 'pyramids' | 'blending';

const TABS: { key: TabKey; label: string; Panel: ComponentType }[] = [
  { key: 'resampling', label: 'Resampling & aliasing', Panel: ResamplingTab },
  { key: 'pyramids', label: 'Pyramids', Panel: PyramidTab },
  { key: 'blending', label: 'Blending', Panel: BlendingTab },
];

/**
 * Three tabs. A tab is mounted the first time it is opened and then kept (hidden), so
 * switching back keeps its settings and uploads without recomputing anything.
 */
export function PyramidsSection() {
  const [tab, setTab] = useState<TabKey>('resampling');
  const [opened, setOpened] = useState<ReadonlySet<TabKey>>(() => new Set<TabKey>(['resampling']));

  const select = (key: TabKey) => {
    setTab(key);
    setOpened(prev => (prev.has(key) ? prev : new Set(prev).add(key)));
  };

  return (
    <div className="demo-section">
      <div className="tab-bar">
        <div className="tab-list" role="tablist" aria-label="Pyramids views">
          {TABS.map(t => (
            <button key={t.key} type="button" role="tab" id={`pyr-tab-${t.key}`} aria-controls={`pyr-panel-${t.key}`}
              aria-selected={tab === t.key} className={`tab-btn${tab === t.key ? ' tab-active' : ''}`} onClick={() => select(t.key)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {TABS.map(({ key, Panel }) => (
        <div key={key} id={`pyr-panel-${key}`} role="tabpanel" aria-labelledby={`pyr-tab-${key}`} hidden={tab !== key}
          className="demo-section pyr-panel">
          {opened.has(key) && <Panel />}
        </div>
      ))}
    </div>
  );
}
