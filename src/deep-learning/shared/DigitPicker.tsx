import { useMemo, useState } from 'react';
import { PixelCanvas } from '../../components/PixelCanvas';
import { useText } from '../../lib/i18n';
import type { MnistSplit } from './mnist';
import { floatImage } from './render';
import './dl-shared.css';

const TEXT = {
  en: { pick: 'Pick a test digit', image: (d: number) => `Test image of a ${d}`, other: 'Other examples' },
  fr: { pick: 'Choisissez un chiffre de test', image: (d: number) => `Image de test d'un ${d}`, other: 'Autres exemples' },
};

interface DigitPickerProps {
  split: MnistSplit;
  /** Index of the selected image in `split`. */
  value: number;
  onChange: (index: number) => void;
  label?: string;
}

/** Indices of `split`'s images grouped by digit. */
function byDigit(split: MnistSplit): number[][] {
  const groups: number[][] = Array.from({ length: 10 }, () => []);
  for (let i = 0; i < split.count; i++) groups[split.labels[i]].push(i);
  return groups;
}

/** One example of each digit from a held-out split, with a button for other examples. */
export function DigitPicker({ split, value, onChange, label: labelProp }: DigitPickerProps) {
  const t = useText(TEXT);
  const label = labelProp ?? t.pick;
  const groups = useMemo(() => byDigit(split), [split]);
  const [page, setPage] = useState(0);
  const choices = groups.map(g => g[page % g.length]);
  const images = useMemo(
    () => groups.map(g => {
      const i = g[page % g.length];
      return floatImage(split.images.subarray(i * 784, (i + 1) * 784), 28, 28);
    }),
    [groups, page, split],
  );

  return (
    <div className="digit-picker" role="group" aria-label={label}>
      <span className="demo-label">{label}:</span>
      <div className="digit-picker-row">
        {choices.map((index, d) => (
          <button
            key={d}
            type="button"
            className={`digit-picker-btn${index === value ? ' digit-picker-active' : ''}`}
            aria-pressed={index === value}
            aria-label={t.image(d)}
            onClick={() => onChange(index)}
          >
            <PixelCanvas image={images[d]} size={36} label="" />
          </button>
        ))}
      </div>
      <button
        type="button"
        className="btn btn-sm"
        onClick={() => {
          const next = page + 1;
          setPage(next);
          const d = split.labels[value];
          onChange(groups[d][next % groups[d].length]);
        }}
      >
        🎲 {t.other}
      </button>
    </div>
  );
}
