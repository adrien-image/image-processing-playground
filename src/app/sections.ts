import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

export const SECTION_KEYS = [
  'convolution', 'segmentation', 'matching', 'color', 'morphology', 'regions', 'edges', 'frequency',
  'pyramids', 'hough', 'histogram', 'augmentation', 'denoising', 'template', 'classification',
] as const;

export type SectionKey = (typeof SECTION_KEYS)[number];

export interface SectionDef {
  key: SectionKey;
  icon: string;
  label: string;
  /** One-line summary for the lesson grid on the home page. */
  blurb: string;
  Component: LazyExoticComponent<ComponentType>;
}

/**
 * Every section is its own lazily-loaded chunk, so visiting one tab only downloads
 * that tab's code (TensorFlow.js is only fetched for Classification).
 */
export const SECTIONS: SectionDef[] = [
  { key: 'convolution', icon: '🖼', label: 'Convolution',
    blurb: 'Slide a kernel over an image and watch every multiply-and-sum: Sobel, Gaussian blur, sharpen.',
    Component: lazy(() => import('../convolution/ConvolutionSection').then(m => ({ default: m.ConvolutionSection }))) },
  { key: 'segmentation', icon: '🎯', label: 'Segmentation',
    blurb: 'Split an image into regions with Otsu thresholding and k-means clustering.',
    Component: lazy(() => import('../segmentation/SegmentationSection').then(m => ({ default: m.SegmentationSection }))) },
  { key: 'matching', icon: '🔗', label: 'Matching',
    blurb: 'Harris corners, descriptors and RANSAC to find one image inside another.',
    Component: lazy(() => import('../matching/MatchingSection').then(m => ({ default: m.MatchingSection }))) },
  { key: 'color', icon: '🎨', label: 'Color',
    blurb: 'RGB and HSV channels, hue, saturation, brightness and gamma.',
    Component: lazy(() => import('../color/ColorSection').then(m => ({ default: m.ColorSection }))) },
  { key: 'morphology', icon: '🔲', label: 'Morphology',
    blurb: 'Erosion, dilation, opening and closing on binary and grayscale images.',
    Component: lazy(() => import('../morphology/MorphologySection').then(m => ({ default: m.MorphologySection }))) },
  { key: 'regions', icon: '🧩', label: 'Regions',
    blurb: 'Label connected components and measure area, centroid, perimeter and circularity.',
    Component: lazy(() => import('../regions/RegionsSection').then(m => ({ default: m.RegionsSection }))) },
  { key: 'edges', icon: '⚡', label: 'Edges',
    blurb: 'Sobel, Prewitt, Laplacian and Canny edge detectors side by side.',
    Component: lazy(() => import('../edges/EdgeComparisonSection').then(m => ({ default: m.EdgeComparisonSection }))) },
  { key: 'frequency', icon: '🌀', label: 'Frequency',
    blurb: 'The 2-D Fourier spectrum, with low-, high- and band-pass filters.',
    Component: lazy(() => import('../frequency/FrequencySection').then(m => ({ default: m.FrequencySection }))) },
  { key: 'pyramids', icon: '🔺', label: 'Pyramids',
    blurb: 'Resampling and aliasing, Gaussian and Laplacian pyramids, multiband blending.',
    Component: lazy(() => import('../pyramids/PyramidsSection').then(m => ({ default: m.PyramidsSection }))) },
  { key: 'hough', icon: '📐', label: 'Hough',
    blurb: 'Find lines and circles by voting in the Hough accumulator.',
    Component: lazy(() => import('../hough/HoughSection').then(m => ({ default: m.HoughSection }))) },
  { key: 'histogram', icon: '📊', label: 'Histogram',
    blurb: 'Spread out the intensities with histogram equalisation.',
    Component: lazy(() => import('../histogram/HistEqSection').then(m => ({ default: m.HistEqSection }))) },
  { key: 'augmentation', icon: '📦', label: 'Augmentation',
    blurb: 'Chain random transformations to grow a training set.',
    Component: lazy(() => import('../augmentation/AugmentationSection').then(m => ({ default: m.AugmentationSection }))) },
  { key: 'denoising', icon: '🧹', label: 'Denoising',
    blurb: 'Median vs. Gaussian filtering on noisy images, scored with PSNR.',
    Component: lazy(() => import('../denoising/DenoisingSection').then(m => ({ default: m.DenoisingSection }))) },
  { key: 'template', icon: '🔍', label: 'Template',
    blurb: 'Find a patch in an image with normalised cross-correlation.',
    Component: lazy(() => import('../template/TemplateSection').then(m => ({ default: m.TemplateSection }))) },
  { key: 'classification', icon: '🧠', label: 'Classification',
    blurb: 'Train a CNN on MNIST in your browser, draw digits and look inside the network.',
    Component: lazy(() => import('../dl/DLPlayground').then(m => ({ default: m.DLPlayground }))) },
];
