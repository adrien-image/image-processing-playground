/** Small deterministic PRNG (mulberry32): same seed → same sequence, so demos are repeatable. */
export function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal sample (mean 0, variance 1) from a uniform generator (Box–Muller). */
export function gaussianRandom(random: () => number): number {
  const u = 1 - random(); // (0, 1]: log(0) would be −∞
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}
