export const DEMO_IMAGES = ['checkerboard', 'circle', 'edges', 'corner'] as const;
export type DemoImage = (typeof DEMO_IMAGES)[number];
