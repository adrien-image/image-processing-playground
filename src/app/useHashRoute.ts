import { useCallback, useSyncExternalStore } from 'react';

const subscribe = (onChange: () => void) => {
  window.addEventListener('hashchange', onChange);
  return () => window.removeEventListener('hashchange', onChange);
};
const getHash = () => window.location.hash;

/**
 * Keeps the current section in the URL hash (`#/edges`), so sections can be
 * bookmarked, shared, refreshed and navigated with the back button. Hash routing
 * works on GitHub Pages without any server rewrites.
 */
export function useHashRoute<K extends string>(keys: readonly K[], fallback: K): [K, (key: K) => void] {
  const hash = useSyncExternalStore(subscribe, getHash);
  const key = hash.replace(/^#\/?/, '');
  const route = (keys as readonly string[]).includes(key) ? (key as K) : fallback;
  const navigate = useCallback((next: K) => {
    window.location.hash = `/${next}`;
  }, []);
  return [route, navigate];
}
