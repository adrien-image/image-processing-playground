import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

export const SECTION_KEYS = [
  'convolution', 'segmentation', 'matching', 'color', 'morphology', 'edges', 'frequency',
  'hough', 'histogram', 'augmentation', 'denoising', 'template', 'classification',
] as const;

export type SectionKey = (typeof SECTION_KEYS)[number];

export interface SectionDef {
  key: SectionKey;
  icon: string;
  label: string;
  Component: LazyExoticComponent<ComponentType>;
}

/**
 * Every section is its own lazily-loaded chunk, so visiting one tab only downloads
 * that tab's code (TensorFlow.js is only fetched for Classification).
 */
export const SECTIONS: SectionDef[] = [
  { key: 'convolution', icon: '🖼', label: 'Convolution',
    Component: lazy(() => import('../convolution/ConvolutionSection').then(m => ({ default: m.ConvolutionSection }))) },
  { key: 'segmentation', icon: '🎯', label: 'Segmentation',
    Component: lazy(() => import('../segmentation/SegmentationSection').then(m => ({ default: m.SegmentationSection }))) },
  { key: 'matching', icon: '🔗', label: 'Matching',
    Component: lazy(() => import('../matching/MatchingSection').then(m => ({ default: m.MatchingSection }))) },
  { key: 'color', icon: '🎨', label: 'Color',
    Component: lazy(() => import('../color/ColorSection').then(m => ({ default: m.ColorSection }))) },
  { key: 'morphology', icon: '🔲', label: 'Morphology',
    Component: lazy(() => import('../morphology/MorphologySection').then(m => ({ default: m.MorphologySection }))) },
  { key: 'edges', icon: '⚡', label: 'Edges',
    Component: lazy(() => import('../edges/EdgeComparisonSection').then(m => ({ default: m.EdgeComparisonSection }))) },
  { key: 'frequency', icon: '🌀', label: 'Frequency',
    Component: lazy(() => import('../frequency/FrequencySection').then(m => ({ default: m.FrequencySection }))) },
  { key: 'hough', icon: '📐', label: 'Hough',
    Component: lazy(() => import('../hough/HoughSection').then(m => ({ default: m.HoughSection }))) },
  { key: 'histogram', icon: '📊', label: 'Histogram',
    Component: lazy(() => import('../histogram/HistEqSection').then(m => ({ default: m.HistEqSection }))) },
  { key: 'augmentation', icon: '📦', label: 'Augmentation',
    Component: lazy(() => import('../augmentation/AugmentationSection').then(m => ({ default: m.AugmentationSection }))) },
  { key: 'denoising', icon: '🧹', label: 'Denoising',
    Component: lazy(() => import('../denoising/DenoisingSection').then(m => ({ default: m.DenoisingSection }))) },
  { key: 'template', icon: '🔍', label: 'Template',
    Component: lazy(() => import('../template/TemplateSection').then(m => ({ default: m.TemplateSection }))) },
  { key: 'classification', icon: '🧠', label: 'Classification',
    Component: lazy(() => import('../dl/DLPlayground').then(m => ({ default: m.DLPlayground }))) },
];
