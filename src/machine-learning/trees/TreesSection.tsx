import { useCallback, useDeferredValue, useMemo, useState, type ReactNode } from 'react';
import treesCode from './trees.ts?raw';
import pythonCode from './trees.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { LineChart } from '../../components/LineChart';
import { seededRandom } from '../../lib/random';
import { makeClassification, SHAPE_LABELS, subset, type Shape2D } from '../shared/datasets';
import { CLASS_COLORS, classMap, UNIT_EXTENT } from '../shared/plot';
import { ScatterPlot, type Draw } from '../shared/ScatterPlot';
import {
  accuracyAtDepths, decisionPath, fitClassificationTree, fitForest, forestProba, predictProba, splitRows, treeSize,
  type Impurity, type TreeNode,
} from './trees';
import '../shared/ml-shared.css';
import './trees.css';

const N = 400;
const MAX_DEPTH = 12;
const MAX_TREES = 100;
const SHOWN_TREES = 4;
const DEFAULT_NOISE: Record<Shape2D, number> = { blobs: 0.15, moons: 0.25, circles: 0.15, xor: 0.15, spiral: 0.05 };
const FEATURE = ['x', 'y'];

const fmt = (v: number, digits = 2) => v.toFixed(digits).replace('-', '−');
const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

/** Split lines of a tree cut at maxDepth, each drawn across the cell it splits; thicker near the root. */
function drawSplits(d: Draw, tree: TreeNode, maxDepth: number, alpha = 1) {
  const { ctx, toPx, color } = d;
  ctx.strokeStyle = color(`color-mix(in srgb, var(--fg) ${Math.round(70 * alpha)}%, transparent)`);
  const walk = (node: TreeNode, x0: number, x1: number, y0: number, y1: number) => {
    if (!node.split || node.depth >= maxDepth) return;
    const { feature, threshold: t, left, right } = node.split;
    ctx.lineWidth = Math.max(0.75, 2.5 - 0.4 * node.depth) * alpha;
    ctx.beginPath();
    const [a, b] = feature === 0 ? [toPx(t, y0), toPx(t, y1)] : [toPx(x0, t), toPx(x1, t)];
    ctx.moveTo(...a); ctx.lineTo(...b);
    ctx.stroke();
    if (feature === 0) { walk(left, x0, t, y0, y1); walk(right, t, x1, y0, y1); }
    else { walk(left, x0, x1, y0, t); walk(right, x0, x1, t, y1); }
  };
  walk(tree, UNIT_EXTENT.x[0], UNIT_EXTENT.x[1], UNIT_EXTENT.y[0], UNIT_EXTENT.y[1]);
}

/** The cell of the partition that contains (x, y), for a tree cut at maxDepth. */
function leafCell(tree: TreeNode, x: number, y: number, maxDepth: number): [number, number, number, number] {
  let [x0, x1, y0, y1] = [UNIT_EXTENT.x[0], UNIT_EXTENT.x[1], UNIT_EXTENT.y[0], UNIT_EXTENT.y[1]];
  let node = tree;
  while (node.split && node.depth < maxDepth) {
    const { feature, threshold: t } = node.split;
    const goLeft = (feature === 0 ? x : y) <= t;
    if (feature === 0) { if (goLeft) x1 = t; else x0 = t; } else if (goLeft) y1 = t; else y0 = t;
    node = goLeft ? node.split.left : node.split.right;
  }
  return [x0, x1, y0, y1];
}

export function TreesSection() {
  const [shape, setShape] = useState<Shape2D>('moons');
  const [noise, setNoise] = useState(DEFAULT_NOISE.moons);
  const [seed, setSeed] = useState(1);
  const [impurity, setImpurity] = useState<Impurity>('gini');
  const [depth, setDepth] = useState(5);
  const [minLeaf, setMinLeaf] = useState(1);
  const [mode, setMode] = useState<'tree' | 'forest'>('tree');
  const [nTrees, setNTrees] = useState(30);
  const [picked, setPicked] = useState<[number, number] | null>(null);
  const [showCode, setShowCode] = useState(false);
  // Refitting 100 trees takes a few tens of milliseconds: let the noise slider move first.
  const lazyNoise = useDeferredValue(noise);

  const { train, test, classes } = useMemo(() => {
    const all = makeClassification(shape, N, lazyNoise, seededRandom(seed), 3);
    const rows = splitRows(N, 0.5, seededRandom(seed + 1000));
    return { train: subset(all, rows.train), test: subset(all, rows.test), classes: all.classes };
  }, [shape, lazyNoise, seed]);

  // One deep tree and one forest of 100 deep trees: every depth and forest size is a cut or a prefix of these.
  const tree = useMemo(() => fitClassificationTree(train.xy, 2, train.labels, classes, { maxDepth: MAX_DEPTH, minLeaf, impurity }),
    [train, classes, minLeaf, impurity]);
  const forest = useMemo(() => fitForest(train.xy, 2, train.labels, classes,
    { maxDepth: MAX_DEPTH, minLeaf, impurity, trees: MAX_TREES, bootstrap: true, maxFeatures: 1 }, seed), [train, classes, minLeaf, impurity, seed]);
  const grove = useMemo(() => forest.trees.slice(0, nTrees), [forest, nTrees]);

  const curves = useMemo(() => ({
    treeTrain: accuracyAtDepths([tree], train.xy, 2, train.labels, classes, MAX_DEPTH),
    treeTest: accuracyAtDepths([tree], test.xy, 2, test.labels, classes, MAX_DEPTH),
    forestTrain: accuracyAtDepths(grove, train.xy, 2, train.labels, classes, MAX_DEPTH),
    forestTest: accuracyAtDepths(grove, test.xy, 2, test.labels, classes, MAX_DEPTH),
  }), [tree, grove, train, test, classes]);

  const map = useMemo(() => (mode === 'tree'
    ? classMap((x, y) => predictProba(tree, [x, y], depth), classes, 120)
    : classMap((x, y) => forestProba(forest, [x, y], depth, nTrees), classes, 96)), [mode, tree, forest, depth, nTrees, classes]);
  const smallMaps = useMemo(() => (mode === 'forest'
    ? forest.trees.slice(0, SHOWN_TREES).map(t => classMap((x, y) => predictProba(t, [x, y], depth), classes, 60)) : []), [mode, forest, depth, classes]);

  const shownTree = mode === 'tree' ? tree : forest.trees[0];
  const path = useMemo(() => (picked ? decisionPath(shownTree, picked, depth) : []), [picked, shownTree, depth]);
  const size = treeSize(shownTree, depth);

  const overlay = useCallback((d: Draw) => {
    if (mode === 'tree') {
      if (picked) {
        const [x0, x1, y0, y1] = leafCell(tree, picked[0], picked[1], depth);
        const [ax, ay] = d.toPx(x0, y1), [bx, by] = d.toPx(x1, y0);
        d.ctx.fillStyle = d.color('color-mix(in srgb, var(--fg) 14%, transparent)');
        d.ctx.fillRect(ax, ay, bx - ax, by - ay);
      }
      drawSplits(d, tree, depth);
    }
    if (picked) {
      // A cross with a halo, visible on every class colour.
      const [px, py] = d.toPx(picked[0], picked[1]);
      const cross = () => {
        d.ctx.beginPath();
        d.ctx.moveTo(px - 8, py - 8); d.ctx.lineTo(px + 8, py + 8); d.ctx.moveTo(px - 8, py + 8); d.ctx.lineTo(px + 8, py - 8);
        d.ctx.stroke();
      };
      d.ctx.strokeStyle = d.color('var(--bg)'); d.ctx.lineWidth = 6; cross();
      d.ctx.strokeStyle = d.color('var(--fg)'); d.ctx.lineWidth = 2.5; cross();
    }
  }, [mode, tree, depth, picked]);

  const chooseShape = (s: Shape2D) => { setShape(s); setNoise(DEFAULT_NOISE[s]); setPicked(null); };
  const k = depth - 1;
  const acc = mode === 'tree' ? { train: curves.treeTrain[k], test: curves.treeTest[k] } : { train: curves.forestTrain[k], test: curves.forestTest[k] };
  const bestTreeDepth = curves.treeTest.indexOf(Math.max(...curves.treeTest)) + 1;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <span className="demo-label">Data:</span>
        {(Object.keys(SHAPE_LABELS) as Shape2D[]).map(s => (
          <button key={s} type="button" aria-pressed={shape === s} className={`btn btn-sm${shape === s ? ' btn-primary' : ''}`}
            onClick={() => chooseShape(s)}>{SHAPE_LABELS[s]}</button>
        ))}
        <button type="button" className="btn btn-sm" onClick={() => { setSeed(s => s + 1); setPicked(null); }}>↻ New sample</button>
        <span className="ml-auto">
          <ShowCodeButton code={treesCode} title="trees/trees.ts" python={{ code: pythonCode, title: 'trees/trees.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <span className="demo-label">Model:</span>
        <button type="button" aria-pressed={mode === 'tree'} className={`btn btn-sm${mode === 'tree' ? ' btn-primary' : ''}`} onClick={() => setMode('tree')}>Single tree</button>
        <button type="button" aria-pressed={mode === 'forest'} className={`btn btn-sm${mode === 'forest' ? ' btn-primary' : ''}`} onClick={() => setMode('forest')}>Random forest</button>
        <span className="demo-label">Impurity:</span>
        <button type="button" aria-pressed={impurity === 'gini'} className={`btn btn-sm${impurity === 'gini' ? ' btn-primary' : ''}`} onClick={() => setImpurity('gini')}>Gini</button>
        <button type="button" aria-pressed={impurity === 'entropy'} className={`btn btn-sm${impurity === 'entropy' ? ' btn-primary' : ''}`} onClick={() => setImpurity('entropy')}>Entropy</button>
      </div>

      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Noise <strong>{noise.toFixed(2)}</strong>
            <input type="range" min={0} max={0.5} step={0.01} value={noise} onChange={e => setNoise(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Max depth <strong>{depth}</strong>
            <input type="range" min={1} max={MAX_DEPTH} value={depth} onChange={e => setDepth(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Min samples per leaf <strong>{minLeaf}</strong>
            <input type="range" min={1} max={30} value={minLeaf} onChange={e => setMinLeaf(Number(e.target.value))} />
          </label>
          {mode === 'forest' && (
            <label className="dl-slider">Trees <strong>{nTrees}</strong>
              <input type="range" min={1} max={MAX_TREES} value={nTrees} onChange={e => setNTrees(Number(e.target.value))} />
            </label>
          )}
        </div>
      </div>

      <div className="ml-grid trees-grid">
        <div className="demo-panel">
          <h3>{mode === 'tree' ? 'Partition of a single tree' : `Averaged vote of ${nTrees} tree${nTrees > 1 ? 's' : ''}`} · click a point</h3>
          <ScatterPlot map={map} overlay={overlay} onPick={(x, y) => setPicked([x, y])}
            label={mode === 'tree' ? `Decision map of a tree of depth ${depth} with its axis-aligned split lines` : `Decision map of a random forest of ${nTrees} trees`}
            layers={[
              { xy: train.xy, labels: train.labels, size: 6 },
              { xy: test.xy, labels: test.labels, size: 2.5, shape: 'ring' },
            ]} />
          <div className="ml-legend">
            <span><span className="ml-swatch trees-dot" />training point</span>
            <span><span className="ml-swatch trees-ring" />test point</span>
            {mode === 'tree' && <span>lines: splits, thicker near the root</span>}
          </div>
          <div className="ml-stats">
            <span>train accuracy</span><strong>{pct(acc.train)}</strong>
            <span>test accuracy</span><strong>{pct(acc.test)}</strong>
            <span>{mode === 'tree' ? 'leaves' : 'leaves of tree 1'}</span><strong>{size.leaves}</strong>
            <span>depth reached</span><strong>{size.depth}</strong>
          </div>
        </div>

        <div className="demo-panel">
          <h3>Accuracy against max depth</h3>
          <LineChart title="Train and test accuracy of a single tree and of the forest, for max depth 1 to 12" xStart={1} xLabel="max depth"
            markerX={k} yMax={1} formatY={v => `${Math.round(100 * v)}%`} height={240} xTickLabel={i => String(i + 1)}
            series={[
              { label: 'tree · train', color: 'var(--series-orange)', values: curves.treeTrain, dashed: true },
              { label: 'tree · test', color: 'var(--series-orange)', values: curves.treeTest },
              { label: `forest (${nTrees}) · train`, color: 'var(--series-blue)', values: curves.forestTrain, dashed: true },
              { label: `forest (${nTrees}) · test`, color: 'var(--series-blue)', values: curves.forestTest },
            ]} />
          <p className="caption">The tree's training accuracy climbs to 100% as it memorises; its test accuracy peaks at depth {bestTreeDepth} ({pct(curves.treeTest[bestTreeDepth - 1])})
            and ends at {pct(curves.treeTest[MAX_DEPTH - 1])}. The forest of the same deep trees ends at {pct(curves.forestTest[MAX_DEPTH - 1])}.</p>
        </div>

        <div className="demo-panel trees-wide">
          <h3>{mode === 'tree' ? 'The tree' : 'Tree 1 of the forest'}: first four levels</h3>
          <TreeDiagram tree={shownTree} maxDepth={depth} path={path} impurity={impurity} />
          {picked ? (
            <p className="trees-path">
              <span className="muted">Point ({fmt(picked[0])}, {fmt(picked[1])}):</span>{' '}
              {path.slice(0, -1).map((node, i) => {
                const s = node.split!;
                const left = path[i + 1] === s.left;
                return <span key={i}>{FEATURE[s.feature]} {left ? '≤' : '>'} {fmt(s.threshold)} → </span>;
              })}
              <strong>leaf: <span className="ml-swatch" style={{ background: CLASS_COLORS[path[path.length - 1].value] }} />
                class {path[path.length - 1].value + 1}</strong> (training points per class: {path[path.length - 1].counts.join(' / ')})
            </p>
          ) : <p className="caption">Click the map: the point's path through the tree is highlighted, and listed here down to its leaf.</p>}
        </div>

        {mode === 'forest' && (
          <div className="demo-panel trees-wide">
            <h3>Four trees of the forest</h3>
            <div className="trees-small">
              {smallMaps.map((m, t) => (
                <figure key={t}>
                  <ScatterPlot map={m} size={170} label={`Decision map of tree ${t + 1} of the forest, on its bootstrap sample`}
                    overlay={d => drawSplits(d, forest.trees[t], depth, 0.6)}
                    layers={[{ xy: train.xy, labels: train.labels, size: 3.5, mask: forest.inBag[t] }]} />
                  <figcaption className="caption">tree {t + 1} · {forest.inBag[t].filter(c => c > 0).length} of {train.n} points drawn</figcaption>
                </figure>
              ))}
            </div>
            <p className="caption">Each tree sees its own bootstrap sample (only the points it drew are shown) and chooses each split
              from one feature picked at random, so the trees overfit in different places. Their average is smoother than any of them.</p>
          </div>
        )}
      </div>

      <div className="demo-explanation">
        <h4>Twenty questions, asked greedily</h4>
        <p>A decision tree classifies a point by asking one question per node, "is x ≤ 0.31?", and following the answer down to a
        leaf, which votes with the classes of the training points that ended there. Training is greedy: at each node every feature
        and every threshold between two consecutive training values is tried, and the split that lowers the impurity most is kept,
        then each child is split the same way:</p>
        <p className="formula"><code>gain = I(parent) − (n_L/n)·I(left) − (n_R/n)·I(right),  Gini I = 1 − Σ p_c²,  entropy I = −Σ p_c log₂ p_c</code></p>
        <p>Both impurities are zero for a pure node and largest for an even mix; they almost always pick the same splits (toggle them and
        watch the map barely change). Each question looks at one coordinate only, so the partition is made of <strong>axis-aligned
        rectangles</strong>: a diagonal or curved boundary becomes a staircase, and the steps only get finer with more data and depth.
        The same property means trees <strong>need no feature scaling</strong>: stretching an axis moves the thresholds but not the
        partition of the training points.</p>
        <p><strong>Depth is memorisation.</strong> Every extra level can halve the cells, so a deep enough tree isolates every training
        point: with one point per leaf allowed, training accuracy reaches 100% whatever the noise, and the map grows little islands
        around noisy points. On the default two moons (noise 0.25, 200 training and 200 test points) the single tree's test accuracy
        peaks at depth 2 (89.5%) and falls to 82.5% at depth 12, where it gets every training point right; raise <em>min samples
        per leaf</em> to stop the islands, or lower the depth.</p>
        <p>A <strong>random forest</strong> keeps the deep trees but averages many of them. Each grows on a bootstrap sample (n draws with
        replacement, about 63% of the distinct points) and picks each split from a randomly chosen feature, so the trees make
        <em> different</em> mistakes. Averaging B trees with variance σ² and correlation ρ leaves ρσ² + (1 − ρ)σ²/B: the
        decorrelation is what makes averaging pay. Its training accuracy still climbs towards 100%, but its test curve no longer
        falls with depth: 30 trees of depth 12 score 88.0% on the same test points, against 82.5% for one. Its map has soft, smoother edges where the trees disagree. Gradient boosting (next lesson) combines trees the
        other way round: shallow trees, added one at a time, each correcting the last.</p>
      </div>
    </div>
  );
}

const BOX_W = 112, BOX_H = 52, SLOT = 124, LEVEL = 82, LEVELS = 4;

/** The first levels of a tree as SVG boxes (split, impurity, counts), with a point's path highlighted. */
function TreeDiagram({ tree, maxDepth, path, impurity }: { tree: TreeNode; maxDepth: number; path: TreeNode[]; impurity: Impurity }) {
  const levels = Math.min(LEVELS, maxDepth + 1);
  const slots = 2 ** (levels - 1);
  const width = slots * SLOT, height = levels * LEVEL - (LEVEL - BOX_H) + 18;
  const onPath = new Set(path);
  const boxes: ReactNode[] = [], edges: React.ReactNode[] = [];
  const place = (node: TreeNode, pos: number) => {
    const span = width / 2 ** node.depth;
    const cx = (pos + 0.5) * span, top = node.depth * LEVEL;
    const split = node.depth < maxDepth ? node.split : undefined;
    const lit = onPath.has(node);
    if (split && node.depth < levels - 1) {
      for (const [child, p] of [[split.left, 2 * pos], [split.right, 2 * pos + 1]] as const) {
        const ccx = (p + 0.5) * (width / 2 ** (node.depth + 1));
        edges.push(<line key={`e${node.depth}-${p}`} className={`trees-edge${lit && onPath.has(child) ? ' trees-lit' : ''}`}
          x1={cx} y1={top + BOX_H} x2={ccx} y2={top + LEVEL} />);
        if (node.depth === 0) edges.push(<text key={`l${p}`} className="trees-edge-label" x={(cx + ccx) / 2 + (p === 0 ? -8 : 8)}
          y={top + BOX_H + 18} textAnchor={p === 0 ? 'end' : 'start'}>{p === 0 ? 'yes' : 'no'}</text>);
        place(child, p);
      }
    }
    let x0 = cx - BOX_W / 2;
    const bar = node.counts.map((c, i) => {
      const w = ((BOX_W - 16) * c) / node.n;
      const r = <rect key={i} x={x0 + 8} y={top + 38} width={w} height={6} style={{ fill: CLASS_COLORS[i] }} />;
      x0 += w;
      return r;
    });
    const hidden = split && node.depth === levels - 1 ? treeSize(node, maxDepth).leaves : 0;
    boxes.push(
      <g key={`n${node.depth}-${pos}`} className={`trees-node${split ? '' : ' trees-leaf'}${lit ? ' trees-lit' : ''}`}>
        <rect x={cx - BOX_W / 2} y={top} width={BOX_W} height={BOX_H} rx={6} />
        <text x={cx} y={top + 15} className="trees-node-title" textAnchor="middle">
          {split ? `${FEATURE[split.feature]} ≤ ${fmt(split.threshold)}` : `leaf → class ${node.value + 1}`}
        </text>
        <text x={cx} y={top + 30} className="trees-node-info" textAnchor="middle">{impurity} {node.impurity.toFixed(2)} · n {node.n}</text>
        {bar}
        {hidden > 0 && <text x={cx} y={top + BOX_H + 13} className="trees-node-info" textAnchor="middle">▾ {hidden} leaves below</text>}
      </g>,
    );
  };
  place(tree, 0);
  return (
    <div className="trees-diagram">
      <svg viewBox={`0 0 ${width} ${height}`} style={{ minWidth: Math.min(width, 560) }} role="img"
        aria-label={`The first ${levels} levels of the tree: split, ${impurity} and number of training points at each node`}>
        {edges}
        {boxes}
      </svg>
    </div>
  );
}
