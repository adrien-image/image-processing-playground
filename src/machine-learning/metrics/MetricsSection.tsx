import { useMemo, useState, type CSSProperties, type PointerEvent } from 'react';
import metricsCode from './metrics.ts?raw';
import pythonCode from './metrics.py?raw';
import { ShowCodeButton } from '../../components/CodeViewer';
import { CLASS_COLORS } from '../shared/plot';
import {
  averagePrecision, cheapestThreshold, confusionAt, prCurve, ratesOf, rocAuc, rocCurve, scoredTestSet, sweep,
  type CurvePoint,
} from './metrics';
import '../shared/ml-shared.css';
import './metrics.css';

const BINS = 40;
const NEG = CLASS_COLORS[0], POS = CLASS_COLORS[1];

const pct = (v: number, digits = 1) => (Number.isFinite(v) ? `${(100 * v).toFixed(digits)}%` : '—');
const num = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—');

export function MetricsSection() {
  const [rate, setRate] = useState(0.1);
  const [separation, setSeparation] = useState(2);
  const [threshold, setThreshold] = useState(0.5);
  const [costRatio, setCostRatio] = useState(10);
  const [perClass, setPerClass] = useState(true);
  const [showCode, setShowCode] = useState(false);

  // Retrained only when the problem changes; the threshold only re-counts.
  const data = useMemo(() => scoredTestSet(rate, separation), [rate, separation]);
  const curves = useMemo(() => {
    const s = sweep(data.scores, data.labels);
    return { s, roc: rocCurve(s), pr: prCurve(s), auc: rocAuc(s), ap: averagePrecision(s) };
  }, [data]);
  const confusion = confusionAt(data.scores, data.labels, threshold);
  const r = ratesOf(confusion);
  const { positives, negatives } = curves.s;
  const positiveRate = positives / (positives + negatives);
  const cheapest = useMemo(() => cheapestThreshold(curves.s, costRatio), [curves, costRatio]);
  const cost = costRatio * confusion.fn + confusion.fp;

  return (
    <div className="demo-section">
      <div className="demo-controls">
        <div className="dl-sliders">
          <label className="dl-slider">Positives <strong>{pct(rate, 0)}</strong>
            <input type="range" min={0.02} max={0.5} step={0.01} value={rate} onChange={e => setRate(Number(e.target.value))} />
          </label>
          <label className="dl-slider">Class separation d′ <strong>{separation.toFixed(1)}</strong>
            <input type="range" min={0.5} max={4} step={0.1} value={separation} onChange={e => setSeparation(Number(e.target.value))} />
          </label>
        </div>
        <span className="ml-auto">
          <ShowCodeButton code={metricsCode} title="metrics/metrics.ts" python={{ code: pythonCode, title: 'metrics/metrics.py' }} open={showCode} onOpenChange={setShowCode} />
        </span>
      </div>

      <div className="demo-controls">
        <label className="dl-slider met-threshold">Threshold t <strong>{threshold.toFixed(3)}</strong>
          <input type="range" min={0} max={1} step={0.005} value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
        </label>
        <button type="button" className="btn btn-sm" onClick={() => setThreshold(0.5)}>t = 0.5</button>
        <label className="dl-slider">A miss costs <strong>{costRatio}×</strong> a false alarm
          <input type="range" min={1} max={50} step={1} value={costRatio} onChange={e => setCostRatio(Number(e.target.value))} />
        </label>
        <button type="button" className="btn btn-sm btn-primary" onClick={() => setThreshold(Math.min(1, cheapest.threshold))}>
          Cheapest threshold ({Number.isFinite(cheapest.threshold) ? cheapest.threshold.toFixed(3) : 'none'})
        </button>
        <span className="dl-status">cost here {cost} · cheapest {cheapest.cost}</span>
      </div>

      <div className="ml-grid met-grid">
        <div className="demo-panel">
          <h3>Test scores by class · click to set t</h3>
          <Histogram scores={data.scores} labels={data.labels} threshold={threshold} perClass={perClass} onPick={setThreshold} />
          <div className="ml-legend">
            <span><span className="ml-swatch" style={{ background: NEG }} />negatives ({negatives})</span>
            <span><span className="ml-swatch" style={{ background: POS }} />positives ({positives})</span>
            <button type="button" className="btn btn-sm" aria-pressed={perClass} onClick={() => setPerClass(p => !p)}>
              {perClass ? 'Scaled per class' : 'True counts (√ scale)'}
            </button>
          </div>
          <p className="caption">Logistic regression trained on 2000 points, scored on 2000 fresh ones. Everything right of the
          line is called positive. Scaled per class, each histogram fills the height; true counts show the imbalance.</p>
        </div>

        <div className="demo-panel">
          <h3>Confusion matrix at t = {threshold.toFixed(3)}</h3>
          <div className="met-confusion" role="table" aria-label="Confusion matrix">
            <div role="row" className="met-row">
              <span role="columnheader" className="met-head met-corner">actual ╲ predicted</span>
              <span role="columnheader" className="met-head">predicted +</span>
              <span role="columnheader" className="met-head">predicted −</span>
            </div>
            <div role="row" className="met-row">
              <span role="rowheader" className="met-head">actual +</span>
              <Cell name="TP" count={confusion.tp} total={positives} good note="hits" />
              <Cell name="FN" count={confusion.fn} total={positives} note="misses" />
            </div>
            <div role="row" className="met-row">
              <span role="rowheader" className="met-head">actual −</span>
              <Cell name="FP" count={confusion.fp} total={negatives} note="false alarms" />
              <Cell name="TN" count={confusion.tn} total={negatives} good note="correct rejections" />
            </div>
          </div>
          <table className="met-table">
            <tbody>
              <Metric name="Accuracy" question="How often is it right?" formula="(TP + TN) / N" value={r.accuracy} />
              <Metric name="Precision" question="When it says positive, is it?" formula="TP / (TP + FP)" value={r.precision} />
              <Metric name="Recall (TPR)" question="How many positives does it find?" formula="TP / (TP + FN)" value={r.recall} />
              <Metric name="Specificity" question="How many negatives does it clear?" formula="TN / (TN + FP)" value={r.specificity} />
              <Metric name="F1" question="Precision and recall together" formula="2PR / (P + R)" value={r.f1} />
              <tr className="met-baseline">
                <th scope="row">"Always negative"<span>a model that never says positive</span></th>
                <td className="mono">N₋ / N</td>
                <td className="mono">{pct(1 - positiveRate)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="demo-panel">
          <h3>ROC curve · AUC {curves.auc.toFixed(3)}</h3>
          <CurvePlot points={curves.roc} kind="roc" current={[r.fpr, r.recall]} baseline={0}
            label={`ROC curve with area ${curves.auc.toFixed(3)}; the current threshold gives false-positive rate ${pct(r.fpr)} and true-positive rate ${pct(r.recall)}`} />
          <p className="caption">True-positive rate against false-positive rate, one point per threshold. Dashed: a coin flip (AUC 0.5).</p>
        </div>

        <div className="demo-panel">
          <h3>Precision–recall · AP {curves.ap.toFixed(3)}</h3>
          <CurvePlot points={curves.pr} kind="pr" current={Number.isFinite(r.precision) ? [r.recall, r.precision] : null} baseline={positiveRate}
            label={`Precision against recall with average precision ${curves.ap.toFixed(3)}; the no-skill baseline is ${pct(positiveRate)}`} />
          <p className="caption">Dashed: the no-skill baseline, precision = the positive rate ({pct(positiveRate)}). The shaded
          staircase is the average precision.</p>
        </div>
      </div>

      <div className="demo-explanation">
        <h4>One score, many metrics</h4>
        <p>A classifier rarely says "yes" or "no": it gives a score, and someone picks the <strong>threshold</strong>. Every
        point of the histogram right of the line is called positive; orange points there are true positives, blue ones false
        alarms, and orange points left of it are misses. Slide the threshold and the four counts trade places: there is no
        setting that lowers both kinds of mistake, only a choice of which to make. Each metric asks a different question
        of those counts (the table), and the right one depends on what a mistake costs.</p>
        <p><strong>Accuracy misleads under imbalance.</strong> With the default 10% positives and d′ = 2, the model at t = 0.5
        is 92.3% accurate, against 90% for a "model" that always answers negative, yet it finds only 43% of the positives.
        Slide the positives down to 2%: accuracy at t = 0.5 is 98.0%, exactly the always-negative score, with a recall of 15%.
        Precision and recall, which ignore the mass of easy true negatives, show what is going on.</p>
        <p className="formula"><code>AUC = P(score of a random positive &gt; score of a random negative)    AP = Σₙ (Rₙ − Rₙ₋₁)·Pₙ</code></p>
        <p><strong>ROC or PR?</strong> The ROC curve plots the hit rate against the false-alarm rate; each is a fraction of one
        class only, so the curve does not depend on how common the classes are. At d′ = 2 the AUC is 0.921 with 50% positives
        and 0.920 with 2%. Its area has a clean meaning: the probability that the model ranks a random positive above a
        random negative. The PR curve puts precision on the y axis, which mixes the classes: with few positives, even a small
        false-alarm rate produces many false alarms relative to the hits. The same model's AP falls from 0.922 to 0.362
        between those two settings. Use ROC to compare how well models <em>rank</em>, PR (and AP) when positives are rare and
        what matters is how trustworthy a positive call is: fraud, disease screening, search results. The no-skill PR
        baseline is the positive rate, not 0.5.</p>
        <p><strong>Choosing the threshold is a cost decision.</strong> If a miss costs k times a false alarm, the cheapest
        threshold minimises k·FN + FP. For calibrated probabilities (logistic regression's roughly are) the theory says
        call positive when p &gt; 1/(1 + k): k = 10 gives about 0.09, far below 0.5. Press "Cheapest threshold" to find it from
        the test counts. None of these numbers should be tuned on the test set in practice: pick the threshold on a
        validation split, then report the test metrics once.</p>
      </div>
    </div>
  );
}

function Cell({ name, count, total, good, note }: { name: string; count: number; total: number; good?: boolean; note: string }) {
  const share = total ? count / total : 0;
  return (
    <span role="cell" className={`met-cell ${good ? 'met-good' : 'met-bad'}`} style={{ '--share': `${Math.round(8 + 40 * share)}%` } as CSSProperties}>
      <span className="met-cell-name">{name}</span>
      <strong>{count}</strong>
      <span className="met-cell-note">{note} · {pct(share)} of row</span>
    </span>
  );
}

function Metric({ name, question, formula, value }: { name: string; question: string; formula: string; value: number }) {
  return (
    <tr>
      <th scope="row">{name}<span>{question}</span></th>
      <td className="mono">{formula}</td>
      <td className="mono">{num(value)}</td>
    </tr>
  );
}

// ---- SVG plots (theme colours come straight from CSS variables) ----

const W = 360, H = 210, PAD = { left: 34, right: 10, top: 10, bottom: 30 };

function Histogram({ scores, labels, threshold, perClass, onPick }: {
  scores: ArrayLike<number>; labels: ArrayLike<number>; threshold: number; perClass: boolean; onPick: (t: number) => void;
}) {
  const bins = useMemo(() => {
    const neg = new Array<number>(BINS).fill(0), pos = new Array<number>(BINS).fill(0);
    for (let i = 0; i < labels.length; i++) {
      const b = Math.min(BINS - 1, Math.floor(scores[i] * BINS));
      (labels[i] ? pos : neg)[b]++;
    }
    return { neg, pos };
  }, [scores, labels]);
  const plotW = W - PAD.left - PAD.right, plotH = H - PAD.top - PAD.bottom;
  const maxNeg = Math.max(1, ...bins.neg), maxPos = Math.max(1, ...bins.pos), maxAll = Math.max(maxNeg, maxPos);
  const x = (v: number) => PAD.left + v * plotW;
  // Per class: each histogram fills the height. True counts: one shared √ scale, so a rare class stays visible.
  const bar = (counts: number[], max: number, color: string) => counts.map((c, b) => {
    const h = (perClass ? c / max : Math.sqrt(c / max)) * plotH;
    return c ? <rect key={b} x={x(b / BINS) + 0.5} width={plotW / BINS - 1} y={PAD.top + plotH - h} height={h} style={{ fill: color }} className="met-bar" /> : null;
  });
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = ((e.clientX - rect.left) / rect.width) * W;
    onPick(Math.round(Math.min(1, Math.max(0, (fx - PAD.left) / plotW)) * 200) / 200);
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="met-svg met-pick" role="img" onPointerDown={pick}
      aria-label={`Histograms of the test scores of each class, with the threshold at ${threshold.toFixed(3)}`}>
      <rect x={x(Math.min(1, threshold))} y={PAD.top} width={Math.max(0, x(1) - x(Math.min(1, threshold)))} height={plotH} className="met-positive-zone" />
      {bar(bins.pos, perClass ? maxPos : maxAll, POS)}
      {bar(bins.neg, perClass ? maxNeg : maxAll, NEG)}
      <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + plotH} y2={PAD.top + plotH} className="met-axis" />
      {[0, 0.25, 0.5, 0.75, 1].map(t => (
        <text key={t} x={x(t)} y={PAD.top + plotH + 11} className="met-tick" textAnchor="middle">{t}</text>
      ))}
      {!perClass && <text x={PAD.left - 4} y={PAD.top + 8} className="met-tick" textAnchor="end">{maxAll}</text>}
      <text x={PAD.left - 4} y={PAD.top + plotH} className="met-tick" textAnchor="end">0</text>
      <text x={x(0.5)} y={H - 3} className="met-tick met-label" textAnchor="middle">score = P(positive)</text>
      <line x1={x(threshold)} x2={x(threshold)} y1={PAD.top - 4} y2={PAD.top + plotH} className="met-threshold-line" />
    </svg>
  );
}

const S = 260, P = { left: 36, right: 10, top: 10, bottom: 32 };
const side = S - P.left - P.right;
const px = (v: number) => P.left + v * side, py = (v: number) => P.top + (1 - v) * side;

/** A unit-square curve plot: ROC (linear segments) or PR (a staircase, the shape AP measures). */
function CurvePlot({ points, kind, current, baseline, label }: {
  points: CurvePoint[]; kind: 'roc' | 'pr'; current: [number, number] | null; baseline: number; label: string;
}) {
  const path = useMemo(() => {
    if (kind === 'roc') return points.map((p, i) => `${i ? 'L' : 'M'}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`).join('');
    // Precision Pₙ holds over the recall step (Rₙ₋₁, Rₙ]: the rectangles that sum to AP.
    let d = `M${px(0)},${py(points[1]?.y ?? 1)}`;
    for (let k = 1; k < points.length; k++) {
      d += `V${py(points[k].y).toFixed(1)}H${px(points[k].x).toFixed(1)}`;
    }
    return d;
  }, [points, kind]);
  const area = `${path}V${py(0)}H${px(0)}Z`;
  const ticks = [0, 0.5, 1];
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="met-svg met-curve" role="img" aria-label={label}>
      <rect x={px(0)} y={py(1)} width={side} height={side} className="met-frame" />
      {ticks.map(t => (
        <g key={t}>
          <line x1={px(t)} x2={px(t)} y1={py(0)} y2={py(1)} className="met-grid" />
          <line x1={px(0)} x2={px(1)} y1={py(t)} y2={py(t)} className="met-grid" />
          <text x={px(t)} y={py(0) + 12} className="met-tick" textAnchor="middle">{t}</text>
          <text x={px(0) - 4} y={py(t) + 3} className="met-tick" textAnchor="end">{t}</text>
        </g>
      ))}
      <text x={px(0.5)} y={S - 4} className="met-tick met-label" textAnchor="middle">{kind === 'roc' ? 'false-positive rate' : 'recall'}</text>
      <text x={10} y={py(0.5)} className="met-tick met-label" textAnchor="middle" transform={`rotate(-90 10 ${py(0.5)})`}>
        {kind === 'roc' ? 'true-positive rate (recall)' : 'precision'}
      </text>
      {kind === 'roc'
        ? <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} className="met-chance" />
        : <line x1={px(0)} y1={py(baseline)} x2={px(1)} y2={py(baseline)} className="met-chance" />}
      <path d={area} className="met-area" />
      <path d={path} className="met-line" />
      {current && <circle cx={px(current[0])} cy={py(current[1])} r={5} className="met-point" />}
    </svg>
  );
}
