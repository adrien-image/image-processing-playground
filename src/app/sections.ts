import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

export const SECTION_KEYS = [
  'convolution', 'segmentation', 'matching', 'color', 'morphology', 'regions', 'edges', 'frequency', 'jpeg',
  'pyramids', 'hough', 'histogram', 'augmentation', 'denoising', 'deblur', 'template', 'optical-flow', 'stereo', 'stitching', 'classification',
  'learn-kernel', 'receptive-field', 'saliency', 'adversarial', 'regularisation', 'autoencoder', 'unet', 'detection', 'diffusion', 'vit',
] as const;

export type SectionKey = (typeof SECTION_KEYS)[number];

export type GroupKey = 'image-processing' | 'deep-learning';

export interface GroupDef {
  key: GroupKey;
  label: string;
  icon: string;
  blurb: string;
}

export const GROUPS: GroupDef[] = [
  { key: 'image-processing', label: 'Image processing', icon: '🖼',
    blurb: 'Classic algorithms that transform, measure and match pixels: every step is a formula you can follow.' },
  { key: 'deep-learning', label: 'Deep learning', icon: '🧠',
    blurb: 'Networks that learn their filters from data, trained live in your browser, and the tools to see what they learned.' },
];

export interface SectionDef {
  key: SectionKey;
  group: GroupKey;
  icon: string;
  label: string;
  /** One-line summary for the lesson grid on the home page. */
  blurb: string;
  Component: LazyExoticComponent<ComponentType>;
}

/** Sections of one group, in menu order. */
export function sectionsIn(group: GroupKey): SectionDef[] {
  return SECTIONS.filter(s => s.group === group);
}

/**
 * Every section is its own lazily-loaded chunk, so visiting one tab only downloads
 * that tab's code (TensorFlow.js is only fetched for Classification).
 */
export const SECTIONS: SectionDef[] = [
  { key: 'convolution', group: 'image-processing', icon: '🖼', label: 'Convolution',
    blurb: 'Slide a kernel over an image and watch every multiply-and-sum: Sobel, Gaussian blur, sharpen.',
    Component: lazy(() => import('../image-processing/convolution/ConvolutionSection').then(m => ({ default: m.ConvolutionSection }))) },
  { key: 'segmentation', group: 'image-processing', icon: '🎯', label: 'Segmentation',
    blurb: 'Split an image into regions with Otsu thresholding and k-means clustering.',
    Component: lazy(() => import('../image-processing/segmentation/SegmentationSection').then(m => ({ default: m.SegmentationSection }))) },
  { key: 'matching', group: 'image-processing', icon: '🔗', label: 'Matching',
    blurb: 'Harris corners, descriptors and RANSAC to find one image inside another.',
    Component: lazy(() => import('../image-processing/matching/MatchingSection').then(m => ({ default: m.MatchingSection }))) },
  { key: 'color', group: 'image-processing', icon: '🎨', label: 'Color',
    blurb: 'RGB and HSV channels, hue, saturation, brightness and gamma.',
    Component: lazy(() => import('../image-processing/color/ColorSection').then(m => ({ default: m.ColorSection }))) },
  { key: 'morphology', group: 'image-processing', icon: '🔲', label: 'Morphology',
    blurb: 'Erosion, dilation, opening and closing on binary and grayscale images.',
    Component: lazy(() => import('../image-processing/morphology/MorphologySection').then(m => ({ default: m.MorphologySection }))) },
  { key: 'regions', group: 'image-processing', icon: '🧩', label: 'Regions',
    blurb: 'Label connected components and measure area, centroid, perimeter and circularity.',
    Component: lazy(() => import('../image-processing/regions/RegionsSection').then(m => ({ default: m.RegionsSection }))) },
  { key: 'edges', group: 'image-processing', icon: '⚡', label: 'Edges',
    blurb: 'Sobel, Prewitt, Laplacian and Canny edge detectors side by side.',
    Component: lazy(() => import('../image-processing/edges/EdgeComparisonSection').then(m => ({ default: m.EdgeComparisonSection }))) },
  { key: 'frequency', group: 'image-processing', icon: '🌀', label: 'Frequency',
    blurb: 'The 2-D Fourier spectrum, with low-, high- and band-pass filters.',
    Component: lazy(() => import('../image-processing/frequency/FrequencySection').then(m => ({ default: m.FrequencySection }))) },
  { key: 'jpeg', group: 'image-processing', icon: '🗜', label: 'JPEG',
    blurb: 'How JPEG shrinks images: 8×8 DCT, quantisation and zigzag run-lengths, with a block-by-block inspector.',
    Component: lazy(() => import('../image-processing/jpeg/JpegSection').then(m => ({ default: m.JpegSection }))) },
  { key: 'pyramids', group: 'image-processing', icon: '🔺', label: 'Pyramids',
    blurb: 'Resampling and aliasing, Gaussian and Laplacian pyramids, multiband blending.',
    Component: lazy(() => import('../image-processing/pyramids/PyramidsSection').then(m => ({ default: m.PyramidsSection }))) },
  { key: 'hough', group: 'image-processing', icon: '📐', label: 'Hough',
    blurb: 'Find lines and circles by voting in the Hough accumulator.',
    Component: lazy(() => import('../image-processing/hough/HoughSection').then(m => ({ default: m.HoughSection }))) },
  { key: 'histogram', group: 'image-processing', icon: '📊', label: 'Histogram',
    blurb: 'Spread out the intensities with histogram equalisation.',
    Component: lazy(() => import('../image-processing/histogram/HistEqSection').then(m => ({ default: m.HistEqSection }))) },
  { key: 'augmentation', group: 'image-processing', icon: '📦', label: 'Augmentation',
    blurb: 'Chain random transformations to grow a training set.',
    Component: lazy(() => import('../image-processing/augmentation/AugmentationSection').then(m => ({ default: m.AugmentationSection }))) },
  { key: 'denoising', group: 'image-processing', icon: '🧹', label: 'Denoising',
    blurb: 'Median vs. Gaussian filtering on noisy images, scored with PSNR.',
    Component: lazy(() => import('../image-processing/denoising/DenoisingSection').then(m => ({ default: m.DenoisingSection }))) },
  { key: 'deblur', group: 'image-processing', icon: '🔬', label: 'Deblurring',
    blurb: 'Undo motion blur and defocus: the inverse filter, the Wiener filter and Richardson–Lucy, and why noise makes it hard.',
    Component: lazy(() => import('../image-processing/deblur/DeblurSection').then(m => ({ default: m.DeblurSection }))) },
  { key: 'template', group: 'image-processing', icon: '🔍', label: 'Template',
    blurb: 'Find a patch in an image with normalised cross-correlation.',
    Component: lazy(() => import('../image-processing/template/TemplateSection').then(m => ({ default: m.TemplateSection }))) },
  { key: 'optical-flow', group: 'image-processing', icon: '🌊', label: 'Optical flow',
    blurb: 'Track how every part of the image moves between two frames: Lucas–Kanade, the aperture problem and pyramids.',
    Component: lazy(() => import('../image-processing/optical-flow/OpticalFlowSection').then(m => ({ default: m.OpticalFlowSection }))) },
  { key: 'stereo', group: 'image-processing', icon: '👓', label: 'Stereo depth',
    blurb: 'Depth from two cameras: block matching along each row, window size trade-offs, occlusions and the left–right check.',
    Component: lazy(() => import('../image-processing/stereo/StereoSection').then(m => ({ default: m.StereoSection }))) },
  { key: 'stitching', group: 'image-processing', icon: '🧷', label: 'Panorama stitching',
    blurb: 'Match corners between two photos, fit a homography with RANSAC, warp one onto the other and blend the seam.',
    Component: lazy(() => import('../image-processing/stitching/StitchingSection').then(m => ({ default: m.StitchingSection }))) },
  { key: 'classification', group: 'deep-learning', icon: '🧠', label: 'Classification',
    blurb: 'Train a CNN on MNIST in your browser, draw digits and look inside the network.',
    Component: lazy(() => import('../deep-learning/classification/DLPlayground').then(m => ({ default: m.DLPlayground }))) },
  { key: 'learn-kernel', group: 'deep-learning', icon: '🎓', label: 'Learn a kernel',
    blurb: 'Gradient descent finds a convolution kernel from examples: watch the loss fall and the weights converge.',
    Component: lazy(() => import('../deep-learning/learn-kernel/LearnKernelSection').then(m => ({ default: m.LearnKernelSection }))) },
  { key: 'receptive-field', group: 'deep-learning', icon: '🔭', label: 'Receptive field',
    blurb: 'Which input pixels can a unit deep in a CNN see? Kernel size, stride, pooling and dilation, layer by layer.',
    Component: lazy(() => import('../deep-learning/receptive-field/ReceptiveFieldSection').then(m => ({ default: m.ReceptiveFieldSection }))) },
  { key: 'saliency', group: 'deep-learning', icon: '🔦', label: 'Saliency',
    blurb: 'Which pixels drove the prediction? Gradients, SmoothGrad, Grad-CAM and occlusion side by side.',
    Component: lazy(() => import('../deep-learning/saliency/SaliencySection').then(m => ({ default: m.SaliencySection }))) },
  { key: 'adversarial', group: 'deep-learning', icon: '🎭', label: 'Adversarial',
    blurb: 'Fool a trained network with changes too small to see: FGSM, targeted and iterative attacks.',
    Component: lazy(() => import('../deep-learning/adversarial/AdversarialSection').then(m => ({ default: m.AdversarialSection }))) },
  { key: 'regularisation', group: 'deep-learning', icon: '🛡', label: 'Overfitting',
    blurb: 'Watch a network memorise its training set, then fight back with dropout, weight decay, augmentation and more data.',
    Component: lazy(() => import('../deep-learning/regularisation/RegularisationSection').then(m => ({ default: m.RegularisationSection }))) },
  { key: 'autoencoder', group: 'deep-learning', icon: '🗜', label: 'Autoencoder',
    blurb: 'Squeeze every digit into two numbers and explore the latent space: plain vs. variational autoencoder.',
    Component: lazy(() => import('../deep-learning/autoencoder/AutoencoderSection').then(m => ({ default: m.AutoencoderSection }))) },
  { key: 'unet', group: 'deep-learning', icon: '🧬', label: 'U-Net',
    blurb: 'Train a small U-Net to segment shapes where a global threshold fails, with and without skip connections.',
    Component: lazy(() => import('../deep-learning/unet/UNetSection').then(m => ({ default: m.UNetSection }))) },
  { key: 'detection', group: 'deep-learning', icon: '📦', label: 'Detection',
    blurb: 'How detectors turn thousands of boxes into answers: IoU, non-max suppression, precision–recall and anchors.',
    Component: lazy(() => import('../deep-learning/detection/DetectionSection').then(m => ({ default: m.DetectionSection }))) },
  { key: 'diffusion', group: 'deep-learning', icon: '🌫', label: 'Diffusion',
    blurb: 'How image generators work: destroy data with noise step by step, then train a network to undo it.',
    Component: lazy(() => import('../deep-learning/diffusion/DiffusionSection').then(m => ({ default: m.DiffusionSection }))) },
  { key: 'vit', group: 'deep-learning', icon: '🔀', label: 'Vision Transformer',
    blurb: 'Patches become tokens and attend to each other: train a tiny ViT and see where it looks.',
    Component: lazy(() => import('../deep-learning/vit/ViTSection').then(m => ({ default: m.ViTSection }))) },
];
