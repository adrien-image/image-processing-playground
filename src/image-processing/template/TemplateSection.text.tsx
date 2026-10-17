const en = {
  heading: 'Template Matching (NCC) — click Scene A, or focus it and use the arrow keys + Enter, to pick a template',
  sceneAPick: 'Scene A — pick template',
  sceneAAria: (x: number, y: number) => `Scene A. Template at ${x}, ${y}. Arrow keys move, Enter selects.`,
  sceneA: 'Scene A',
  template: (w: number, h: number) => `Template (${w}×${h})`,
  templateLabel: 'Selected template',
  clickSceneA: 'Click Scene A',
  sceneBFind: 'Scene B — find match',
  sceneB: 'Scene B',
  scoreMap: 'NCC score map',
  scoreLabel: 'NCC score at every template position',
  flatPlaceholder: 'No score map for a flat patch',
  pickPlaceholder: 'Pick a template',
  scoreCaption: 'Black = −1 · white = +1 (best match ringed)',
  matchAt: 'Match at',
  nccScore: 'NCC score:',
  flat: (
    <>
      ⚠ That patch is flat — it has no texture, so NCC is undefined (it would match every flat area equally).
      Pick a patch containing an edge or corner.
    </>
  ),
  explanation: (pw: number, ph: number) => (
    <>
      <h4>Template Matching — Normalised Cross-Correlation</h4>
      <p><strong>Pick a patch in Scene A</strong> to extract a {pw}×{ph} template.
      The algorithm slides it over every position in <strong>Scene B</strong> — a different image where the shapes are at new positions — and scores each one.</p>
      <p>The <span className="text-success">green rectangle</span> marks the best score; the score map shows how strongly every position matched.
      NCC subtracts the mean and divides by the standard deviation of both the template and the image window, so a shape that is uniformly darker or lighter in Scene B still scores ~1.</p>
      <p>Because of that normalisation, a perfectly flat patch can't be matched: its standard deviation is zero.</p>
    </>
  ),
};

const fr: typeof en = {
  heading: 'Mise en correspondance de gabarit (NCC) : cliquez sur la scène A, ou donnez-lui le focus et utilisez les flèches + Entrée, pour choisir un gabarit',
  sceneAPick: 'Scène A : choisir le gabarit',
  sceneAAria: (x: number, y: number) => `Scène A. Gabarit en ${x}, ${y}. Les flèches le déplacent, Entrée le sélectionne.`,
  sceneA: 'Scène A',
  template: (w: number, h: number) => `Gabarit (${w}×${h})`,
  templateLabel: 'Gabarit sélectionné',
  clickSceneA: 'Cliquez sur la scène A',
  sceneBFind: 'Scène B : trouver la correspondance',
  sceneB: 'Scène B',
  scoreMap: 'Carte des scores NCC',
  scoreLabel: 'Score NCC à chaque position du gabarit',
  flatPlaceholder: 'Pas de carte des scores pour un patch uniforme',
  pickPlaceholder: 'Choisissez un gabarit',
  scoreCaption: 'Noir = −1 · blanc = +1 (meilleure correspondance cerclée)',
  matchAt: 'Correspondance en',
  nccScore: 'Score NCC :',
  flat: (
    <>
      ⚠ Ce patch est uniforme : il n'a aucune texture, donc la NCC n'est pas définie (il correspondrait également à toutes les zones uniformes).
      Choisissez un patch qui contient un bord ou un coin.
    </>
  ),
  explanation: (pw: number, ph: number) => (
    <>
      <h4>Mise en correspondance de gabarit : corrélation croisée normalisée</h4>
      <p><strong>Choisissez un patch dans la scène A</strong> pour en extraire un gabarit de {pw}×{ph}.
      L'algorithme le fait glisser sur chaque position de la <strong>scène B</strong> (une autre image, où les formes ont changé de place) et attribue un score à chacune.</p>
      <p>Le <span className="text-success">rectangle vert</span> marque le meilleur score ; la carte des scores montre à quel point chaque position correspond.
      La NCC soustrait la moyenne et divise par l'écart type, à la fois du gabarit et de la fenêtre de l'image : une forme uniformément plus sombre ou plus claire dans la scène B obtient donc quand même un score d'environ 1.</p>
      <p>À cause de cette normalisation, un patch parfaitement uniforme ne peut pas être mis en correspondance : son écart type est nul.</p>
    </>
  ),
};

export const TEXT = { en, fr };
