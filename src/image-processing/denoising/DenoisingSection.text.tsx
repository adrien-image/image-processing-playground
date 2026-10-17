const en = {
  noise: 'Noise:',
  saltPepper: 'Salt & pepper',
  gaussian: 'Gaussian',
  newNoise: '🎲 New noise',
  upload: '📁 Upload',
  demoImage: 'Demo image',
  density: 'Noise density:',
  noiseSigma: 'Noise σ:',
  window: 'Filter window:',
  original: 'Original',
  originalLabel: 'Original grayscale image',
  noisy: (type: 'salt-pepper' | 'gaussian') => `Noisy (${type === 'salt-pepper' ? 'salt & pepper' : 'Gaussian'})`,
  noisyLabel: 'Noisy image',
  median: (size: number) => `Median ${size}×${size}`,
  medianLabel: 'Median-filtered image',
  gaussianTitle: (size: number, sigma: string) => `Gaussian ${size}×${size} (σ=${sigma})`,
  gaussianLabel: 'Gaussian-filtered image',
  explanation: (
    <>
      <h4>Median vs Gaussian</h4>
      <p>The <strong>median filter</strong> replaces each pixel with the median of its neighbourhood. An isolated black or white speck is an outlier,
      so it never becomes the median — salt-and-pepper noise disappears while edges stay crisp.</p>
      <p>The <strong>Gaussian filter</strong> takes a weighted <em>average</em>. Averages are pulled by outliers, so specks become gray smudges;
      but for Gaussian (sensor) noise, where every pixel is slightly off, averaging is exactly what helps.</p>
      <p>Both filters see the same noisy image. <strong>PSNR</strong> (peak signal-to-noise ratio) measures how close each result is to the
      original — higher is better. Switch the noise type to see which filter wins each case; larger windows remove more noise but erase small details.</p>
    </>
  ),
};

const fr: typeof en = {
  noise: 'Bruit :',
  saltPepper: 'Poivre et sel',
  gaussian: 'Gaussien',
  newNoise: '🎲 Nouveau bruit',
  upload: '📁 Charger',
  demoImage: 'Image de démo',
  density: 'Densité du bruit :',
  noiseSigma: 'σ du bruit :',
  window: 'Fenêtre du filtre :',
  original: 'Original',
  originalLabel: 'Image originale en niveaux de gris',
  noisy: (type: 'salt-pepper' | 'gaussian') => `Bruitée (${type === 'salt-pepper' ? 'poivre et sel' : 'gaussien'})`,
  noisyLabel: 'Image bruitée',
  median: (size: number) => `Médian ${size}×${size}`,
  medianLabel: 'Image filtrée par le filtre médian',
  gaussianTitle: (size: number, sigma: string) => `Gaussien ${size}×${size} (σ=${sigma})`,
  gaussianLabel: 'Image filtrée par le filtre gaussien',
  explanation: (
    <>
      <h4>Médian contre gaussien</h4>
      <p>Le <strong>filtre médian</strong> remplace chaque pixel par la médiane de son voisinage. Un point noir ou blanc isolé est une valeur aberrante :
      il ne devient donc jamais la médiane. Le bruit poivre et sel disparaît tandis que les bords restent nets.</p>
      <p>Le <strong>filtre gaussien</strong> calcule une <em>moyenne</em> pondérée. Les moyennes sont tirées par les valeurs aberrantes, si bien que les points
      deviennent des taches grises ; mais pour un bruit gaussien (celui du capteur), où chaque pixel est un peu faux, moyenner est exactement ce qui aide.</p>
      <p>Les deux filtres voient la même image bruitée. Le <strong>PSNR</strong> (rapport signal sur bruit de crête) mesure à quel point chaque résultat est
      proche de l'original : plus il est élevé, mieux c'est. Changez le type de bruit pour voir quel filtre l'emporte dans chaque cas ; des fenêtres plus
      grandes retirent davantage de bruit mais effacent les petits détails.</p>
    </>
  ),
};

export const TEXT = { en, fr };
