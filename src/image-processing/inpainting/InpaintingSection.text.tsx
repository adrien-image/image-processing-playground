const en = {
  damage: 'Damage:',
  presets: { scratch: 'Scratch', box: 'Missing block', text: 'Watermark' } as Record<'scratch' | 'box' | 'text', string>,
  clear: 'Clear (paint your own)',
  brush: 'Brush',
  upload: '📁 Upload',
  damaged: 'Damaged (paint on it)',
  damagedLabel: 'The image with the hole to fill in pink; drag to paint more',
  missing: (n: string) => `${n} missing pixels. Drag to paint more.`,
  diffusion: 'Diffusion',
  diffusionLabel: 'Hole filled by diffusion',
  sweeps: 'Sweeps',
  rmsError: 'RMS error in the hole:',
  exemplar: 'Exemplar-based (Criminisi)',
  exemplarLabel: 'Hole filled by copying patches',
  progress: 'Progress',
  edgesFirst: 'Edges first (priority)',
  source: 'source',
  target: 'target',
  explanation: (
    <>
      <h4>Filling holes</h4>
      <p>Removing a scratch, a watermark or an unwanted object leaves a hole, and the only information about what belongs there is
      what surrounds it. <strong>Diffusion</strong> makes each missing pixel the average of its four neighbours, sweep after sweep,
      until nothing changes: colour flows in from the border like heat. The result is as smooth as possible, which is right for sky
      and wrong for edges and texture: watch the stripes and the box edge turn to mush. Set the sweeps low to see the fill creep in.</p>
      <p className="formula"><code>u(x, y) ← ¼ · [u(x−1, y) + u(x+1, y) + u(x, y−1) + u(x, y+1)]</code></p>
      <p><strong>Exemplar-based</strong> inpainting copies instead of averaging. It takes a patch on the edge of the hole, searches the
      image for the known patch that best matches the part it can see, and pastes the rest. Drag the progress slider back to watch it
      patch by patch (pink box: the patch being filled; cyan: where it copies from). The <strong>order</strong> matters: Criminisi et al.
      fill first where many pixels are already known (confidence) and where an edge runs into the hole (data term), so lines get
      continued before flat areas close over them. Untick "edges first" to fill in plain scan order and compare.</p>
      <p>Modern tools (content-aware fill, diffusion-model inpainting) use the same two ideas, matching and propagation, with
      learned priors about what images look like.</p>
    </>
  ),
};

const fr: typeof en = {
  damage: 'Dégât :',
  presets: { scratch: 'Rayure', box: 'Bloc manquant', text: 'Filigrane' },
  clear: 'Effacer (peindre le vôtre)',
  brush: 'Pinceau',
  upload: '📁 Charger',
  damaged: 'Endommagée (peignez dessus)',
  damagedLabel: 'L\'image avec le trou à combler en rose ; faites glisser pour peindre davantage',
  missing: (n: string) => `${n} pixels manquants. Faites glisser pour en peindre davantage.`,
  diffusion: 'Diffusion',
  diffusionLabel: 'Trou comblé par diffusion',
  sweeps: 'Passes',
  rmsError: 'Erreur RMS dans le trou :',
  exemplar: 'Par exemples (Criminisi)',
  exemplarLabel: 'Trou comblé en copiant des patchs',
  progress: 'Progression',
  edgesFirst: "Bords d'abord (priorité)",
  source: 'source',
  target: 'cible',
  explanation: (
    <>
      <h4>Combler des trous</h4>
      <p>Retirer une rayure, un filigrane ou un objet indésirable laisse un trou, et la seule information sur ce qui devrait s'y
      trouver est ce qui l'entoure. La <strong>diffusion</strong> fait de chaque pixel manquant la moyenne de ses quatre voisins,
      passe après passe, jusqu'à ce que plus rien ne change : la couleur afflue depuis le bord comme de la chaleur. Le résultat est
      aussi lisse que possible, ce qui convient au ciel mais pas aux bords ni aux textures : regardez les bandes et le bord du bloc
      se transformer en bouillie. Réglez peu de passes pour voir le remplissage progresser.</p>
      <p className="formula"><code>u(x, y) ← ¼ · [u(x−1, y) + u(x+1, y) + u(x, y−1) + u(x, y+1)]</code></p>
      <p>La retouche <strong>par exemples</strong> copie au lieu de moyenner. Elle prend un patch au bord du trou, cherche dans
      l'image le patch connu qui correspond le mieux à la partie visible, et colle le reste. Ramenez le curseur de progression en
      arrière pour la regarder travailler patch par patch (cadre rose : le patch en cours de remplissage ; cyan : l'endroit d'où
      elle copie). L'<strong>ordre</strong> compte : Criminisi et al. remplissent d'abord là où beaucoup de pixels sont déjà
      connus (confiance) et là où un bord entre dans le trou (terme de données), si bien que les lignes sont prolongées avant que
      les zones uniformes ne se referment dessus. Décochez « bords d'abord » pour remplir dans le simple ordre de balayage et
      comparer.</p>
      <p>Les outils modernes (remplissage selon le contenu, retouche par modèles de diffusion) reposent sur les deux mêmes idées,
      la mise en correspondance et la propagation, avec des a priori appris sur l'apparence des images.</p>
    </>
  ),
};

export const TEXT = { en, fr };
