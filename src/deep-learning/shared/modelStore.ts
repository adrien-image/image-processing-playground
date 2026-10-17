import { useSyncExternalStore } from 'react';
import type * as tf from '@tensorflow/tfjs';
import type { Texts } from '../../lib/i18n';
import { snapshotLeNet } from '../classification/lenet';

/**
 * The most recently trained LeNet, shared between lessons: Classification publishes
 * its weights after every epoch, and Adversarial and Saliency attack or explain them.
 * The store keeps its own copy of the weights, so the publisher can keep training,
 * reset or dispose its model without affecting the lessons reading this one.
 */
export interface SharedLeNet {
  model: tf.LayersModel;
  /** Changes whenever the weights do, so views can recompute. */
  version: number;
  /** Where the weights came from, for display. */
  source: Texts<string>;
}

let current: SharedLeNet | null = null;
const listeners = new Set<() => void>();

export function publishLeNet(model: tf.LayersModel, source: Texts<string>): void {
  const copy = snapshotLeNet(model, current?.model);
  current = { model: copy, version: (current?.version ?? 0) + 1, source };
  listeners.forEach(l => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useSharedLeNet(): SharedLeNet | null {
  return useSyncExternalStore(subscribe, () => current);
}
