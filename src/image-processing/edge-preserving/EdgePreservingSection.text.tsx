const en = {
  heading: 'Smooth the noise, keep the edges',
  newNoise: '🎲 New noise',
  upload: '📁 Upload',
  demoImage: 'Demo image',
  noiseSigma: 'Noise σ',
  bilateralSs: 'Bilateral σs',
  bilateralSr: 'Bilateral σr',
  nlmH: 'NLM h',
  patch: 'Patch',
  search: 'Search',
  noisyClick: 'Noisy · click a pixel',
  noisyLabel: "Noisy image; click to inspect the filters' weights at a pixel",
  gaussian: (n: number) => `Gaussian ${n}×${n}`,
  gaussianLabel: 'Gaussian-filtered image',
  bilateral: 'Bilateral',
  bilateralLabel: 'Bilateral-filtered image',
  nlm: 'Non-local means',
  nlmLabel: 'Non-local means result',
  bilateralWeights: (x: number, y: number) => `Bilateral weights at (${x}, ${y})`,
  bilateralWeightsLabel: 'Bilateral filter weights in the 7×7 window',
  bilateralCaption: '7×7 window · bright = large weight',
  nlmWeights: 'Non-local means weights',
  nlmWeightsLabel: 'Non-local means weights in the search window',
  nlmCaption: (n: number) => `${n}×${n} search window (the box on the noisy image)`,
  explanation: (
    <>
      <h4>Average only with what looks alike</h4>
      <p>A Gaussian blur gives each neighbour a weight that depends on distance alone, so near an edge it averages the two
      sides and the edge turns into a ramp. The <strong>bilateral filter</strong> multiplies that weight by a second
      Gaussian of the <em>value</em> difference:</p>
      <p className="formula"><code>w(p, q) = exp(−‖p − q‖² / 2σs²) · exp(−(I(q) − I(p))² / 2σr²)</code></p>
      <p>Pixels across an edge differ by much more than σr, so they get no weight: click next to an edge and the weight map
      is cut in two. But it compares single, noisy pixels: σr must be well above the noise σ, or the filter keeps the noise
      too. With σr huge it is a Gaussian blur again.</p>
      <p><strong>Non-local means</strong> compares whole patches instead, which averages the noise out of the comparison,
      and looks further away. The weight of a candidate q is</p>
      <p className="formula"><code>w(p, q) = exp(−max(d²(P(p), P(q)) − 2σ², 0) / h²)</code></p>
      <p>where d² is the mean squared difference of the two patches and 2σ² is what noise alone puts between two copies of
      the same patch. Click a stripe: the weights light up the <em>other</em> stripes, because a repeated texture is
      averaged with its own copies. That is why non-local means usually wins on PSNR, at a much higher cost:
      every pixel compares (2S+1)² patches.</p>
    </>
  ),
};

const fr: typeof en = {
  heading: 'Lisser le bruit, garder les bords',
  newNoise: '🎲 Nouveau bruit',
  upload: '📁 Charger',
  demoImage: 'Image de démo',
  noiseSigma: 'σ du bruit',
  bilateralSs: 'Bilatéral σs',
  bilateralSr: 'Bilatéral σr',
  nlmH: 'NLM h',
  patch: 'Patch',
  search: 'Recherche',
  noisyClick: 'Bruitée · cliquez un pixel',
  noisyLabel: 'Image bruitée ; cliquez pour inspecter les poids des filtres en un pixel',
  gaussian: (n: number) => `Gaussien ${n}×${n}`,
  gaussianLabel: 'Image filtrée par le filtre gaussien',
  bilateral: 'Bilatéral',
  bilateralLabel: 'Image filtrée par le filtre bilatéral',
  nlm: 'Moyennes non locales',
  nlmLabel: 'Résultat des moyennes non locales',
  bilateralWeights: (x: number, y: number) => `Poids bilatéraux en (${x}, ${y})`,
  bilateralWeightsLabel: 'Poids du filtre bilatéral dans la fenêtre 7×7',
  bilateralCaption: 'Fenêtre 7×7 · clair = poids élevé',
  nlmWeights: 'Poids des moyennes non locales',
  nlmWeightsLabel: 'Poids des moyennes non locales dans la fenêtre de recherche',
  nlmCaption: (n: number) => `Fenêtre de recherche ${n}×${n} (le cadre sur l'image bruitée)`,
  explanation: (
    <>
      <h4>Ne moyenner qu'avec ce qui se ressemble</h4>
      <p>Un flou gaussien donne à chaque voisin un poids qui ne dépend que de la distance : près d'un bord, il moyenne donc
      les deux côtés et le bord devient une rampe. Le <strong>filtre bilatéral</strong> multiplie ce poids par une seconde
      gaussienne de l'écart de <em>valeur</em> :</p>
      <p className="formula"><code>w(p, q) = exp(−‖p − q‖² / 2σs²) · exp(−(I(q) − I(p))² / 2σr²)</code></p>
      <p>Les pixels situés de l'autre côté d'un bord diffèrent de bien plus que σr : ils ne reçoivent aucun poids. Cliquez à
      côté d'un bord et la carte des poids est coupée en deux. Mais le filtre compare des pixels isolés et bruités : σr doit
      être nettement supérieur au σ du bruit, sinon le filtre garde aussi le bruit. Avec un σr immense, on retrouve un flou
      gaussien.</p>
      <p>Les <strong>moyennes non locales</strong> comparent plutôt des patchs entiers, ce qui moyenne le bruit hors de la
      comparaison, et cherchent plus loin. Le poids d'un candidat q vaut</p>
      <p className="formula"><code>w(p, q) = exp(−max(d²(P(p), P(q)) − 2σ², 0) / h²)</code></p>
      <p>où d² est l'écart quadratique moyen entre les deux patchs et 2σ² ce que le bruit seul met entre deux copies du même
      patch. Cliquez sur une bande : les poids allument les <em>autres</em> bandes, car une texture répétée est moyennée avec
      ses propres copies. C'est pourquoi les moyennes non locales l'emportent généralement au PSNR, pour un coût bien plus
      élevé : chaque pixel compare (2S+1)² patchs.</p>
    </>
  ),
};

export const TEXT = { en, fr };
