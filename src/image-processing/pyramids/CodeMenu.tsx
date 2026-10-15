import { useId, useState } from 'react';
import { ShowCodeButton } from '../../components/CodeViewer';
import pyramidCode from './pyramid.ts?raw';
import resampleCode from './resample.ts?raw';
import planeCode from './plane.ts?raw';
import demosCode from './demos.ts?raw';

const FILES = {
  'pyramid.ts': pyramidCode,
  'resample.ts': resampleCode,
  'plane.ts': planeCode,
  'demos.ts': demosCode,
} as const;

export type CodeFile = keyof typeof FILES;

/** File picker + "💻 Show code" for the section's algorithm modules. */
export function CodeMenu({ initial }: { initial: CodeFile }) {
  const id = useId();
  const [file, setFile] = useState<CodeFile>(initial);
  const [open, setOpen] = useState(false);
  return (
    <span className="ml-auto pyr-code">
      <label className="visually-hidden" htmlFor={id}>Source file to show</label>
      <select id={id} className="select pyr-code-select" value={file} onChange={e => setFile(e.target.value as CodeFile)}>
        {Object.keys(FILES).map(name => <option key={name} value={name}>{name}</option>)}
      </select>
      <ShowCodeButton code={FILES[file]} title={`pyramids/${file}`} open={open} onOpenChange={setOpen} />
    </span>
  );
}
