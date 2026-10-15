/** Pure helpers for evaluating a classifier's predictions. */

/** cm[true][predicted] = count. */
export function confusionMatrix(trueLabels: ArrayLike<number>, predicted: ArrayLike<number>, classes = 10): number[][] {
  const cm = Array.from({ length: classes }, () => Array<number>(classes).fill(0));
  for (let i = 0; i < trueLabels.length; i++) cm[trueLabels[i]][predicted[i]]++;
  return cm;
}

/**
 * Pick examples to show: up to `correct` right predictions (one per digit where
 * possible) and up to `wrong` mistakes, most confident mistakes first.
 */
export function selectExamples(
  trueLabels: ArrayLike<number>,
  predicted: ArrayLike<number>,
  confidence: ArrayLike<number>,
  { correct = 5, wrong = 5 } = {},
): number[] {
  const right: number[] = [];
  const seenDigits = new Set<number>();
  for (let i = 0; i < trueLabels.length && right.length < correct; i++) {
    if (predicted[i] === trueLabels[i] && !seenDigits.has(trueLabels[i])) {
      seenDigits.add(trueLabels[i]);
      right.push(i);
    }
  }
  // Fewer distinct digits than slots: top up with any other correct predictions.
  for (let i = 0; i < trueLabels.length && right.length < correct; i++) {
    if (predicted[i] === trueLabels[i] && !right.includes(i)) right.push(i);
  }

  const mistakes: number[] = [];
  for (let i = 0; i < trueLabels.length; i++) if (predicted[i] !== trueLabels[i]) mistakes.push(i);
  mistakes.sort((a, b) => confidence[b] - confidence[a]);

  return [...right, ...mistakes.slice(0, wrong)];
}

/**
 * Indices in round-robin order over the labels — first image of each digit 0…9, then
 * the second of each, … — so stepping through a digit-sorted split shows every digit.
 */
export function interleaveByLabel(labels: ArrayLike<number>): number[] {
  const groups = new Map<number, number[]>();
  for (let i = 0; i < labels.length; i++) {
    const group = groups.get(labels[i]);
    if (group) group.push(i);
    else groups.set(labels[i], [i]);
  }
  const ordered = [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, indices]) => indices);
  const order: number[] = [];
  for (let round = 0; order.length < labels.length; round++) {
    for (const indices of ordered) if (round < indices.length) order.push(indices[round]);
  }
  return order;
}
