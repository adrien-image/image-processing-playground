import { useEffect, useMemo, useState } from 'react';
import pcaCode from './pca.ts?raw';
import pythonCode from './pca.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { PixelCanvas } from '../../components/PixelCanvas';
import { DigitPicker } from '../../deep-learning/shared/DigitPicker';
import { floatImage, signedImage } from '../../deep-learning/shared/render';
import { useMnist } from '../../deep-learning/shared/useMnist';
import { seededRandom } from '../../lib/random';
import { componentsFor, decode, encode, explainedVariance, fitPca, project, type Pca } from './pca';
import { DigitLegend, DigitMap } from './DigitMap';
import { sameDigitNeighbours, sampleDigits } from './digits';
import '../shared/ml-shared.css';
import './pca.css';

/** Training digits PCA is fitted on, and how many of them the map shows. */
const N_FIT = 4000;
const N_MAP = 2000;
/** Components computed (enough to reach 95 % of the variance) and power iterations. */
const K = 150;
const ITERATIONS = 6;
const SHOWN_COMPONENTS = 24;
const pct = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}\u00a0%`;

export function PcaSection() {
  const { data, error } = useMnist();
  const [fit, setFit] = useState<{ pca: Pca; seconds: number } | null>(null);
  const [progress, setProgress] = useState(0);
  const [k, setK] = useState(20);
  const [picked, setPicked] = useState(700);
  const [axes, setAxes] = useState<[number, number]>([1, 2]);
  const [showCode, setShowCode] = useState(false);

  const sample = useMemo(() => (data ? sampleDigits(data.train, N_FIT, seededRandom(1)) : null), [data]);

  // Fit in slices, one step of the subspace iteration per animation frame.
  useEffect(() => {
    if (!sample) return;
    const steps = fitPca(sample.images, N_FIT, 784, K, { random: seededRandom(2), iterations: ITERATIONS });
    const start = performance.now();
    let id = 0;
    const tick = () => {
      const s = steps.next();
      if (s.done) {
        setFit({ pca: s.value, seconds: (performance.now() - start) / 1000 });
        return;
      }
      setProgress(s.value);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [sample]);

  const pca = fit?.pca ?? null;
  const variance = useMemo(() => (pca ? explainedVariance(pca) : null), [pca]);
  const keep90 = pca ? componentsFor(pca, 0.9) : null;
  const keep95 = pca ? componentsFor(pca, 0.95) : null;

  const pictures = useMemo(() => {
    if (!pca) return null;
    return {
      mean: floatImage(pca.mean, 28, 28),
      components: Array.from({ length: SHOWN_COMPONENTS }, (_, c) => signedImage(pca.components.subarray(c * 784, (c + 1) * 784), 28, 28)),
    };
  }, [pca]);

  // The picked test digit: its coordinates on all K components, and its residual for every k.
  const digit = useMemo(() => {
    if (!pca || !data) return null;
    const x = data.test.images.subarray(picked * 784, (picked + 1) * 784);
    const z = encode(pca, x);
    let left = 0;
    for (let i = 0; i < 784; i++) left += (x[i] - pca.mean[i]) ** 2;
    const total = left;
    // Components are orthonormal, so each one removes exactly z_c² of the squared error.
    const residual = [1];
    for (let c = 0; c < K; c++) { left -= z[c] * z[c]; residual.push(Math.max(left, 0) / total); }
    return { x, z, total, residual, image: floatImage(x, 28, 28) };
  }, [pca, data, picked]);

  const rebuilt = useMemo(() => {
    if (!pca || !digit) return null;
    const xh = decode(pca, digit.z.subarray(0, k));
    const diff = Float64Array.from(digit.x, (v, i) => v - xh[i]);
    let sq = 0;
    for (const v of diff) sq += v * v;
    return { image: floatImage(xh, 28, 28), diff: signedImage(diff, 28, 28, 1), mse: sq / 784 };
  }, [pca, digit, k]);

  const map = useMemo(() => {
    if (!pca || !sample) return null;
    const xy = project(pca, sample.images.subarray(0, N_MAP * 784), N_MAP, [axes[0] - 1, axes[1] - 1]);
    const labels = sample.labels.subarray(0, N_MAP);
    return { xy, labels, neighbours: sameDigitNeighbours(xy, labels, 5) };
  }, [pca, sample, axes]);

  const ranked = map ? map.neighbours.perDigit.map((v, d) => ({ d, v })).sort((a, b) => b.v - a.v) : [];

  const axisButtons = (which: 0 | 1) => (
    <label className="dl-slider">{which === 0 ? 'x axis: component' : 'y axis: component'} <strong>{axes[which]}</strong>
      <input type="range" min={1} max={10} value={axes[which]}
        onChange={e => { const v = Number(e.target.value); setAxes(a => (which === 0 ? [v, a[1]] : [a[0], v])); }} />
    </label>
  );

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="dl-status">
          {!data ? (error ? '' : 'Loading MNIST…')
            : !fit ? `Computing the covariance and its top ${K} eigenvectors… ${Math.round(100 * progress)}\u00a0%`
            : `${N_FIT.toLocaleString('en-GB')} random training digits · top ${K} components in ${fit.seconds.toFixed(1)} s`}
        </span>
        <span className="ml-auto">
          <ShowCodeButton code={pcaCode} title="pca/pca.ts" python={{ code: pythonCode, title: 'pca/pca.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>
      {error && <div className="error-banner" role="alert">⚠ {error}</div>}
      {!fit && !error && <progress className="pca-progress" value={progress} max={1} aria-label="PCA progress" />}

      {pca && variance && pictures && (
        <div className="ml-grid">
          <div className="demo-panel pca-wide">
            <h3>The mean digit and the first {SHOWN_COMPONENTS} principal components</h3>
            <div className="pca-gallery">
              <figure className="pca-mean">
                <PixelCanvas image={pictures.mean} size={112} label="The mean of the training digits" />
                <figcaption>mean μ</figcaption>
              </figure>
              <div className="pca-components">
                {pictures.components.map((img, c) => (
                  <figure key={c}>
                    <PixelCanvas image={img} size={56} label={`Principal component ${c + 1}`} />
                    <figcaption>v{c + 1} {(100 * variance.ratio[c]).toFixed(1)}%</figcaption>
                  </figure>
                ))}
              </div>
            </div>
            <p className="caption">Each component is a 784-pixel unit vector shown as an image: red pixels positive, blue negative, black
            zero. Under each, the share of the total variance along it. Adding a positive multiple of v1 to the mean digit
            adds the red strokes and removes the blue ones.</p>
          </div>

          <div className="demo-panel">
            <h3>Variance explained</h3>
            <LineChart title="Share of the total variance along each component" xStart={1} height={150}
              series={[{ label: 'per component', color: 'var(--series-blue)', values: variance.ratio.map(r => 100 * r) }]}
              formatY={v => `${v.toFixed(0)}%`} xLabel="component" />
            <LineChart title="Cumulative share of the variance kept by the first k components" xStart={1} height={170} yMin={0} yMax={100}
              series={[
                { label: 'cumulative', color: 'var(--series-green)', values: variance.cumulative.map(r => 100 * r) },
                { label: '90 %', color: 'var(--text-muted)', values: variance.cumulative.map(() => 90), dashed: true },
                { label: '95 %', color: 'var(--text-muted)', values: variance.cumulative.map(() => 95), dashed: true },
              ]}
              formatY={v => `${v.toFixed(0)}%`} xLabel="number of components k" markerX={k - 1} />
            <div className="ml-stats">
              <span>first component</span><strong>{pct(variance.ratio[0])}</strong>
              <span>first {k} (slider)</span><strong>{pct(variance.cumulative[k - 1])}</strong>
              <span>90 % of the variance</span><strong>{keep90 ?? `> ${K}`} components</strong>
              <span>95 % of the variance</span><strong>{keep95 ?? `> ${K}`} components</strong>
            </div>
            <p className="caption">Measured on these {N_FIT.toLocaleString('en-GB')} digits: out of 784 pixels, {keep90} directions hold 90 %
            of the variation.</p>
          </div>

          <div className="demo-panel">
            <h3>Rebuild a test digit from k components</h3>
            <label className="dl-slider">components k <strong>{k}</strong>
              <input type="range" min={1} max={K} value={k} onChange={e => setK(Number(e.target.value))} />
            </label>
            {digit && rebuilt && (
              <>
                <div className="pca-recon">
                  <figure>
                    <PixelCanvas image={digit.image} size={96} label="Original test digit" />
                    <figcaption>original x</figcaption>
                  </figure>
                  <figure>
                    <PixelCanvas image={rebuilt.image} size={96} label={`Reconstruction from ${k} components`} />
                    <figcaption>from {k} components</figcaption>
                  </figure>
                  <figure>
                    <PixelCanvas image={rebuilt.diff} size={96} label="Difference between the original and the reconstruction" />
                    <figcaption>x − x̂</figcaption>
                  </figure>
                </div>
                <LineChart title="Squared error left after k components, as a share of the digit's distance to the mean" height={140} yMin={0} yMax={100}
                  series={[{ label: 'error left', color: 'var(--series-pink)', values: digit.residual.map(r => 100 * r) }]}
                  formatY={v => `${v.toFixed(0)}%`} xLabel="k" markerX={k} />
                <div className="ml-stats">
                  <span>error ‖x − x̂‖² / ‖x − μ‖²</span><strong>{pct(digit.residual[k])}</strong>
                  <span>mean squared error per pixel</span><strong>{rebuilt.mse.toFixed(4)}</strong>
                  <span>numbers stored</span><strong>{k} instead of 784</strong>
                </div>
              </>
            )}
            {data && <DigitPicker split={data.test} value={picked} onChange={setPicked} />}
          </div>

          <div className="demo-panel">
            <h3>{N_MAP.toLocaleString('en-GB')} digits on components {axes[0]} and {axes[1]}</h3>
            {map && <DigitMap xy={map.xy} labels={map.labels} label={`Training digits projected on principal components ${axes[0]} and ${axes[1]}, coloured by digit`} />}
            <DigitLegend />
            <div className="demo-controls pca-axes">
              {axisButtons(0)}
              {axisButtons(1)}
            </div>
            {map && (
              <>
                <p className="caption">How often a point's 5 nearest neighbours on this map are the same digit
                (overall {pct(map.neighbours.overall, 0)}):</p>
                <div className="pca-bars" role="list" aria-label="Same-digit neighbours per digit">
                  {map.neighbours.perDigit.map((v, d) => (
                    <div key={d} role="listitem" className="pca-bar" title={`${d}: ${pct(v, 0)}`}>
                      <span className="pca-bar-track"><span className="pca-bar-fill" style={{ height: `${100 * v}%` }} /></span>
                      <span>{d}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="demo-explanation">
        <h4>Directions of maximum variance</h4>
        <p>An MNIST digit is a point in 784 dimensions, one per pixel, but digits fill only a thin part of that space:
        neighbouring pixels move together, and the corners are always black. PCA finds the directions along which the digits
        actually vary. The first principal component v1 is the unit vector along which the projections (x − μ)·v have the
        largest variance; v2 is the best direction orthogonal to v1, and so on. They are the eigenvectors of the covariance
        matrix, and each eigenvalue is the variance along its direction:</p>
        <p className="formula"><code>C = (1/n) Σ (x − μ)(x − μ)ᵀ,  C·vₖ = λₖ vₖ,  x̂ = μ + Σₖ₌₁..ₖ ((x − μ)·vₖ) vₖ</code></p>
        <p>The covariance here is 784 × 784, and a full eigen-decomposition is slow in a browser. We only need the top {K} directions,
        so the code uses <strong>subspace iteration</strong>: multiply {K + 10} random vectors by C again and again (each multiplication
        stretches the large-variance directions more than the others), re-orthonormalise them with Gram–Schmidt after each
        step, and finish by diagonalising the small matrix VᵀCV.</p>
        <p><strong>Compression.</strong> The reconstruction x̂ stores k numbers instead of 784. Move the slider: a handful of
        components gives a blurry, generic digit, and the residual x − x̂ shows what is still missing. Since the components are
        orthonormal, each one removes exactly ((x − μ)·vₖ)² of the squared error, so the error curve can only go down.
        {keep90 && keep95 && <> On these digits {keep90} components keep 90 % of the variance and {keep95} keep 95 %.</>}</p>
        <p><strong>Why the components look like blurry strokes.</strong> The directions that explain the most variance are the ones
        shared by many digits: v1 contrasts the ring of a 0 (red) with the vertical bar of a 1 (blue): on the map of components 1 and 2,
        0s and 1s sit at opposite ends of the horizontal axis. The next few contrast other frequent stroke patterns. Later
        components are finer and finer patterns, closer to noise. They are not digits, they are the differences between digits.</p>
        <p><strong>Limits.</strong> PCA is linear: it can only rotate and drop axes, and it keeps directions of large variance,
        which are not necessarily the ones that separate classes. On the map of components {axes[0]} and {axes[1]}
        {map && ranked.length === 10 && <>, the {ranked[0].d}s ({pct(ranked[0].v, 0)} same-digit neighbours) and {ranked[1].d}s
        ({pct(ranked[1].v, 0)}) form their own regions, while the {ranked[9].d}s ({pct(ranked[9].v, 0)}) and {ranked[8].d}s
        ({pct(ranked[8].v, 0)}) are lost in the middle</>}{!map && ', a few digits separate and most overlap'}. Try other pairs
        of components: none separates all ten digits. A non-linear map such as t-SNE (next lesson) does much better at keeping
        neighbours together, at the price of a map whose distances mean much less.</p>
      </div>
    </div>
  );
}
