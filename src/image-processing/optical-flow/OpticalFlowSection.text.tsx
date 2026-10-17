import type { Motion } from './opticalFlow';

type Pattern = 'texture' | 'stripes';

const en = {
  motion: 'Motion:',
  motions: { translate: 'translate', rotate: 'rotate', zoom: 'zoom' } as Record<Motion, string>,
  pattern: 'Pattern:',
  patterns: { texture: 'texture', stripes: 'stripes' } as Record<Pattern, string>,
  amount: { translate: 'Shift', rotate: 'Angle', zoom: 'Zoom' } as Record<Motion, string>,
  window: 'Window',
  levels: 'Pyramid levels',
  flick: 'Flick between the frames',
  frame1: 'Frame 1',
  frame2: 'Frame 2',
  firstFrame: 'First frame',
  secondFrame: 'Second frame',
  flicking: 'Flicking every 0.6 s: watch the pattern move.',
  tickFlick: 'Tick "Flick" to see the motion.',
  estimate: 'Lucas–Kanade estimate vs truth',
  estimateLabel: 'Estimated and true motion vectors over the first frame',
  legendGood: 'error < 0.3 px',
  legendOk: '< 1 px',
  legendWrong: 'wrong',
  legendAmbiguous: 'ambiguous',
  legendTruth: 'true motion',
  arrowsScale: (n: number) => `Arrows drawn ${n}× longer than the motion.`,
  meanError: 'Mean error',
  wrongBy: 'Wrong by > 1 px',
  ambiguous: 'Ambiguous',
  explanation: (
    <>
      <h4>Lucas–Kanade optical flow</h4>
      <p>If a patch keeps its brightness while moving by (u, v), then for every pixel of a small window
      <code> Ix·u + Iy·v + It = 0</code>, where Ix, Iy are the image gradients and It the change between frames. One equation per pixel,
      two unknowns: least squares over the window gives a 2×2 system, solved independently at each arrow.</p>
      <p className="formula"><code>[ΣIx² ΣIxIy; ΣIxIy ΣIy²] · (u, v) = −(ΣIxIt, ΣIyIt)</code></p>
      <p>Try the <strong>stripes</strong>: moving along a stripe changes nothing in the image, so only the motion across the stripes can be measured.
      This is the <strong>aperture problem</strong>; the matrix is singular and the arrows turn gray. Then push the <strong>shift</strong> above
      about 7 px: the linear approximation only holds for small motions and the estimates go wrong, until you add <strong>pyramid levels</strong>.
      On a half-size image the motion is half as large; estimate it there, double it, and refine on the next finer level.</p>
      <p>Rotation and zoom move every pixel differently, but the method assumes one motion per window: a bigger window averages more
      (smoother, less noise-sensitive) but blurs the flow field. Deep networks such as RAFT have replaced Lucas–Kanade for dense flow,
      but the brightness-constancy idea is the same.</p>
    </>
  ),
};

const fr: typeof en = {
  motion: 'Mouvement :',
  motions: { translate: 'translation', rotate: 'rotation', zoom: 'zoom' },
  pattern: 'Motif :',
  patterns: { texture: 'texture', stripes: 'bandes' },
  amount: { translate: 'Décalage', rotate: 'Angle', zoom: 'Zoom' },
  window: 'Fenêtre',
  levels: 'Niveaux de pyramide',
  flick: 'Alterner entre les trames',
  frame1: 'Trame 1',
  frame2: 'Trame 2',
  firstFrame: 'Première trame',
  secondFrame: 'Seconde trame',
  flicking: 'Alternance toutes les 0.6 s : regardez le motif bouger.',
  tickFlick: 'Cochez « Alterner » pour voir le mouvement.',
  estimate: 'Estimation de Lucas–Kanade et vérité',
  estimateLabel: 'Vecteurs de mouvement estimés et réels sur la première trame',
  legendGood: 'erreur < 0.3 px',
  legendOk: '< 1 px',
  legendWrong: 'faux',
  legendAmbiguous: 'ambigu',
  legendTruth: 'mouvement réel',
  arrowsScale: (n: number) => `Flèches tracées ${n}× plus longues que le mouvement.`,
  meanError: 'Erreur moyenne',
  wrongBy: 'Faux de > 1 px',
  ambiguous: 'Ambigus',
  explanation: (
    <>
      <h4>Le flot optique de Lucas–Kanade</h4>
      <p>Si un patch garde sa luminosité en se déplaçant de (u, v), alors pour chaque pixel d'une petite fenêtre
      <code> Ix·u + Iy·v + It = 0</code>, où Ix, Iy sont les gradients de l'image et It la variation entre les trames. Une équation par
      pixel, deux inconnues : les moindres carrés sur la fenêtre donnent un système 2×2, résolu indépendamment pour chaque flèche.</p>
      <p className="formula"><code>[ΣIx² ΣIxIy; ΣIxIy ΣIy²] · (u, v) = −(ΣIxIt, ΣIyIt)</code></p>
      <p>Essayez les <strong>bandes</strong> : se déplacer le long d'une bande ne change rien à l'image, donc seul le mouvement
      perpendiculaire aux bandes peut être mesuré. C'est le <strong>problème d'ouverture</strong> ; la matrice est singulière et les
      flèches deviennent grises. Poussez ensuite le <strong>décalage</strong> au-delà d'environ 7 px : l'approximation linéaire ne
      vaut que pour de petits mouvements et les estimations deviennent fausses, jusqu'à ce que vous ajoutiez des <strong>niveaux de
      pyramide</strong>. Sur une image de taille moitié, le mouvement est deux fois plus petit ; on l'estime là, on le double, puis on
      l'affine au niveau plus fin suivant.</p>
      <p>La rotation et le zoom déplacent chaque pixel différemment, mais la méthode suppose un seul mouvement par fenêtre : une
      fenêtre plus grande moyenne davantage (plus lisse, moins sensible au bruit) mais floute le champ de flot. Des réseaux profonds
      comme RAFT ont remplacé Lucas–Kanade pour le flot dense, mais l'idée de la constance de la luminosité reste la même.</p>
    </>
  ),
};

export const TEXT = { en, fr };
