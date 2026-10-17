import type { PsfKind } from './deblur';

const en = {
  blur: 'Blur:',
  kinds: { motion: 'Camera shake (motion)', gaussian: 'Gaussian', defocus: 'Out of focus (disc)' } as Record<PsfKind, string>,
  param: { motion: 'Length (px)', gaussian: 'σ (px)', defocus: 'Radius (px)' } as Record<PsfKind, string>,
  upload: '📁 Upload',
  angle: 'Angle',
  noiseSigma: 'Noise σ',
  wienerK: 'Wiener K',
  rlIterations: 'R–L iterations',
  sharp: 'Sharp original',
  kernel: 'Blur kernel (PSF)',
  kernelLabel: 'The point spread function',
  response: 'Its frequency response |H|',
  responseLabel: "Magnitude of the blur's frequency response, zero frequency in the centre",
  responseCaption: 'Dark = frequencies the blur destroys.',
  blurred: (noisy: boolean) => `Blurred${noisy ? ' + noise' : ''}`,
  inverse: 'Inverse filter G / H',
  wiener: (k: string) => `Wiener, K = ${k}`,
  rl: (n: number) => `Richardson–Lucy, ${n} iterations`,
  explanation: (
    <>
      <h4>Why undoing a blur is hard</h4>
      <p>A blur replaces every point with the kernel's shape: the image is <strong>convolved</strong> with the point spread function.
      In the frequency domain (see the Frequency lesson) that is just a multiplication, G = H · F, so dividing by H should undo it.
      Set the noise to 0 and the <strong>inverse filter</strong> is nearly perfect.</p>
      <p>Add even a little noise and it explodes. The response |H| has frequencies where it is close to zero (the dark stripes and
      rings), and noise divided by almost nothing becomes enormous. The <strong>Wiener filter</strong> divides by |H|² + K instead:
      where the blur kept the signal it inverts it, where it destroyed it, it gives up gracefully. Larger K means less noise but
      less sharpening. <strong>Richardson–Lucy</strong> instead refines an estimate step by step, keeping it positive; run too many
      iterations on a noisy image and it starts sharpening the noise too.</p>
      <p className="formula"><code>F̂ = H* · G / (|H|² + K)</code></p>
      <p>This lesson uses circular convolution, so blur wraps around the image edges. Real photos need their borders handled, and real
      deblurring also has to estimate the PSF itself (blind deconvolution).</p>
    </>
  ),
};

const fr: typeof en = {
  blur: 'Flou :',
  kinds: { motion: 'Bougé (mouvement)', gaussian: 'Gaussien', defocus: 'Mise au point (disque)' },
  param: { motion: 'Longueur (px)', gaussian: 'σ (px)', defocus: 'Rayon (px)' },
  upload: '📁 Charger',
  angle: 'Angle',
  noiseSigma: 'σ du bruit',
  wienerK: 'K de Wiener',
  rlIterations: 'Itérations R–L',
  sharp: 'Original net',
  kernel: 'Noyau de flou (PSF)',
  kernelLabel: "La fonction d'étalement du point",
  response: 'Sa réponse en fréquence |H|',
  responseLabel: 'Module de la réponse en fréquence du flou, fréquence nulle au centre',
  responseCaption: 'Sombre = fréquences détruites par le flou.',
  blurred: (noisy: boolean) => `Flouté${noisy ? ' + bruit' : ''}`,
  inverse: 'Filtre inverse G / H',
  wiener: (k: string) => `Wiener, K = ${k}`,
  rl: (n: number) => `Richardson–Lucy, ${n} itérations`,
  explanation: (
    <>
      <h4>Pourquoi annuler un flou est difficile</h4>
      <p>Un flou remplace chaque point par la forme du noyau : l'image est <strong>convoluée</strong> avec la fonction d'étalement
      du point (PSF). Dans le domaine fréquentiel (voir la leçon Fréquences), ce n'est qu'une multiplication, G = H · F : diviser
      par H devrait donc l'annuler. Réglez le bruit sur 0 et le <strong>filtre inverse</strong> est presque parfait.</p>
      <p>Ajoutez un tout petit peu de bruit et tout explose. La réponse |H| a des fréquences où elle est proche de zéro (les bandes
      et anneaux sombres), et un bruit divisé par presque rien devient énorme. Le <strong>filtre de Wiener</strong> divise plutôt
      par |H|² + K : là où le flou a gardé le signal, il l'inverse ; là où il l'a détruit, il renonce en douceur. Un K plus grand
      donne moins de bruit mais moins de netteté. <strong>Richardson–Lucy</strong>, lui, affine une estimation pas à pas en la
      gardant positive ; avec trop d'itérations sur une image bruitée, il se met à renforcer aussi le bruit.</p>
      <p className="formula"><code>F̂ = H* · G / (|H|² + K)</code></p>
      <p>Cette leçon utilise une convolution circulaire : le flou déborde d'un bord de l'image sur le bord opposé. Les vraies photos
      demandent de traiter leurs bords, et un vrai défloutage doit aussi estimer la PSF elle-même (déconvolution aveugle).</p>
    </>
  ),
};

export const TEXT = { en, fr };
