import type { Texts } from '../lib/i18n';

/** The site's name stays the same in every language; the rest is translated. */
export const SITE_TITLE = 'Image Processing Playground';

export const SITE: Texts<{ tagline: string; description: string }> = {
  en: {
    tagline: 'See how image processing works, one pixel at a time.',
    description:
      'Interactive lessons that run classic computer-vision algorithms on demo images or your own photos, ' +
      'then train neural networks live in your browser: step through a convolution kernel, compare edge detectors, ' +
      'watch gradient descent learn a filter, fool a classifier, explore a latent space or generate with diffusion, ' +
      'fit the classic machine-learning models (SVMs, forests, boosting, PCA, t-SNE), ' +
      'and open the source code behind every result.',
  },
  fr: {
    tagline: 'Voyez comment fonctionne le traitement d\'image, un pixel à la fois.',
    description:
      'Des leçons interactives qui appliquent les algorithmes classiques de vision par ordinateur à des images de démonstration ' +
      'ou à vos propres photos, puis entraînent des réseaux de neurones en direct dans votre navigateur : suivez un noyau de ' +
      'convolution pas à pas, comparez des détecteurs de contours, regardez la descente de gradient apprendre un filtre, ' +
      'trompez un classifieur, explorez un espace latent ou générez par diffusion, ajustez les modèles classiques ' +
      'd\'apprentissage automatique (SVM, forêts, boosting, ACP, t-SNE), et ouvrez le code source derrière chaque résultat.',
  },
};
