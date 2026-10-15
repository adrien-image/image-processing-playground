import { useEffect, useState } from 'react';
import { loadMnist, type MnistData } from './mnist';

let pending: Promise<MnistData> | null = null;

/** MNIST, downloaded once per page and shared by every deep-learning lesson. */
export function getMnist(): Promise<MnistData> {
  pending ??= loadMnist().catch((err: unknown) => {
    pending = null; // let the next caller retry
    throw err;
  });
  return pending;
}

/** Start loading MNIST when `enabled`; `data` is set once it has arrived. */
export function useMnist(enabled = true): { data: MnistData | null; error: string | null } {
  const [state, setState] = useState<{ data: MnistData | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    getMnist().then(
      data => { if (!cancelled) setState({ data, error: null }); },
      (err: unknown) => { if (!cancelled) setState({ data: null, error: err instanceof Error ? err.message : 'Could not load MNIST' }); },
    );
    return () => { cancelled = true; };
  }, [enabled]);
  return state;
}
