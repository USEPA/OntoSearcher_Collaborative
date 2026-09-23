#!/usr/bin/env python3
"""
Quantitative link-prediction evaluation for the paper's Technical Validation.

Evaluates the frozen heterogeneous R-GCN embeddings (nkb_rgcn_embeddings.pt)
on the graph (improved_hetero_data.pt) and reports, per relation type and in
aggregate, the metrics reviewers asked for -- ROC-AUC, PR-AUC, Precision,
Recall, F1, Accuracy, MRR, Hits@K -- each with 95% bootstrap confidence
intervals, plus calibration (Brier score) and conventional-query baselines
(common-neighbors, Adamic-Adar, degree/popularity, random) under an identical
protocol.

Run with the working conda env (base env torch is broken on this machine):

    /Users/pranavsingh/miniforge3/envs/graph_env/bin/python \
        scripts/evaluate_link_prediction.py

Outputs (under results/evaluation/ by default):
    - link_prediction_metrics.json   full nested metrics + config + env
    - link_prediction_summary.md     human-readable summary table
    - fig_auc_by_scorer.png          AUC per scorer (aggregate)
    - fig_pr_by_edgetype.png         PR-AUC per relation (learned scorers)
    - fig_calibration.png            reliability curve (cosine + mlp)
    - fig_score_separation.png       pos vs neg score distributions

This is a TRANSDUCTIVE evaluation of learned embeddings (see module docstring
in src/gnn/evaluation.py); baselines share the protocol so comparisons are fair.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "gnn"))

import evaluation as ev  # noqa: E402


def _env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for mod in ("torch", "torch_geometric", "sklearn", "scipy", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = "not-installed"
    return info


def _aggregate(results, scorer_names, weighted=True):
    """Micro (edge-weighted) and macro (per-type mean) aggregates per scorer."""
    agg = {}
    metric_keys = [
        "roc_auc", "average_precision", "precision", "recall", "f1",
        "accuracy", "best_f1", "mrr", "hits@1", "hits@3", "hits@10",
    ]
    for sn in scorer_names:
        agg[sn] = {}
        for mk in metric_keys:
            vals, weights = [], []
            for r in results:
                if sn in r.scorers and mk in r.scorers[sn]:
                    v = r.scorers[sn][mk]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(v)
                        weights.append(r.num_evaluated)
            if vals:
                agg[sn][f"{mk}_macro"] = float(np.mean(vals))
                if weighted:
                    agg[sn][f"{mk}_micro"] = float(np.average(vals, weights=weights))
    return agg


def make_figures(results, agg, scorer_names, calib, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}

    # 1) AUC per scorer (aggregate macro)
    fig, ax = plt.subplots(figsize=(7, 4))
    aucs = [agg[sn].get("roc_auc_macro", np.nan) for sn in scorer_names]
    ax.bar(scorer_names, aucs, color="#3b7dd8")
    ax.axhline(0.5, ls="--", c="grey", label="random (0.5)")
    ax.set_ylabel("Macro ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Link-prediction ROC-AUC by scorer (mean over relations)")
    for i, v in enumerate(aucs):
        if not np.isnan(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_auc_by_scorer.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs["auc_by_scorer"] = p

    # 2) PR-AUC per edge type for learned scorers
    learned = [s for s in ("mlp", "cosine") if s in scorer_names]
    if learned:
        labels = ["/".join(r.edge_type) for r in results]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 4.5))
        width = 0.8 / max(1, len(learned))
        for j, sn in enumerate(learned):
            vals = [r.scorers.get(sn, {}).get("average_precision", np.nan) for r in results]
            ax.bar(x + j * width, vals, width, label=sn)
        ax.set_xticks(x + width * (len(learned) - 1) / 2)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Average Precision (PR-AUC)")
        ax.set_ylim(0, 1)
        ax.set_title("PR-AUC per relation type")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_pr_by_edgetype.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["pr_by_edgetype"] = p

    # 3) Calibration reliability curve
    if calib:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", c="grey", label="perfect")
        for sn, c in calib.items():
            mp = c.get("reliability_mean_predicted", [])
            fp = c.get("reliability_fraction_positive", [])
            if mp and fp:
                ax.plot(mp, fp, marker="o", label=f"{sn} (Brier {c['brier_post']:.3f})")
        ax.set_xlabel("Mean predicted probability (calibrated)")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title("Reliability curve after calibration")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_calibration.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["calibration"] = p

    # 4) Score separation (pos vs neg means) for learned scorers
    if learned:
        fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.5), 4.5))
        labels = ["/".join(r.edge_type) for r in results]
        x = np.arange(len(labels))
        sn = "mlp" if "mlp" in learned else learned[0]
        pos = [r.scorers.get(sn, {}).get("pos_score_mean", np.nan) for r in results]
        neg = [r.scorers.get(sn, {}).get("neg_score_mean", np.nan) for r in results]
        ax.bar(x - 0.2, pos, 0.4, label="positive", color="#2ca02c")
        ax.bar(x + 0.2, neg, 0.4, label="negative", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel(f"Mean {sn} score")
        ax.set_title(f"Positive vs negative score separation ({sn})")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_score_separation.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["score_separation"] = p

    return figs


def write_summary_md(path, config, env, results, agg, scorer_names, calib):
    lines = []
    lines.append("# Link-Prediction Evaluation Summary\n")
    lines.append(f"_Generated: {env['timestamp_utc']}_\n")
    lines.append("\n## Protocol\n")
    lines.append(
        "- **Task:** predict whether a (source, relation, target) edge exists.\n"
        "- **Positives:** sampled true edges per relation. **Negatives:** filtered "
        "random pairs of the same node types that are not real edges.\n"
        "- **Scorers:** `mlp` = trained R-GCN link-prediction head; `cosine` = "
        "embedding cosine similarity (used for the paper's material-assay table); "
        "plus conventional-graph baselines (`common_neighbors`, `adamic_adar`, "
        "`degree`, `random`).\n"
        "- **Setting:** TRANSDUCTIVE (embeddings trained over the full graph, "
        "standard for KG embeddings). Baselines share the protocol.\n"
        "- **CIs:** 95% via stratified bootstrap "
        f"({config['n_boot']} resamples).\n"
    )
    lines.append("\n## Aggregate metrics (macro mean over relations)\n")
    header = "| Scorer | ROC-AUC | PR-AUC | Precision | Recall | F1 | Best-F1 | MRR | Hits@1 | Hits@3 | Hits@10 |"
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for sn in scorer_names:
        a = agg.get(sn, {})
        def g(k):
            v = a.get(k, float("nan"))
            return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
        lines.append(
            f"| {sn} | {g('roc_auc_macro')} | {g('average_precision_macro')} | "
            f"{g('precision_macro')} | {g('recall_macro')} | {g('f1_macro')} | "
            f"{g('best_f1_macro')} | {g('mrr_macro')} | {g('hits@1_macro')} | "
            f"{g('hits@3_macro')} | {g('hits@10_macro')} |"
        )

    if calib:
        lines.append("\n## Calibration (held-out Brier score; lower is better)\n")
        lines.append("| Scorer | Method | Brier (pre) | Brier (post) |")
        lines.append("|---|---|---|---|")
        for sn, c in calib.items():
            lines.append(f"| {sn} | {c['method']} | {c['brier_pre']:.3f} | {c['brier_post']:.3f} |")

    lines.append("\n## Per-relation ROC-AUC (with 95% CI) - learned scorers\n")
    lines.append("| Relation | n(edges) | mlp AUC [CI] | cosine AUC [CI] |")
    lines.append("|---|---|---|---|")
    for r in results:
        def cell(sn):
            m = r.scorers.get(sn, {})
            if "roc_auc" not in m:
                return "n/a"
            ci = m.get("roc_auc_ci95", [float("nan"), float("nan")])
            return f"{m['roc_auc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        lines.append(
            f"| {'/'.join(r.edge_type)} | {r.num_positives_total} | "
            f"{cell('mlp')} | {cell('cosine')} |"
        )

    lines.append("\n## Limitations\n")
    lines.append(
        "- Transductive setting: absolute AUC/AP are optimistic vs. a fully "
        "inductive split; report relative gains over baselines.\n"
        "- The graph has **no direct material->assay edges**; the paper's "
        "material-assay predictions rely on `cosine` similarity, whose "
        "link-prediction quality is quantified here on the relations that do "
        "exist.\n"
        "- Negatives are randomly sampled (filtered); harder negatives would "
        "lower scores.\n"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=REPO_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "evaluation"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-edges", type=int, default=100, help="Min edges for a relation to be evaluated.")
    ap.add_argument("--max-types", type=int, default=25, help="Max relations to evaluate (by count).")
    ap.add_argument("--sample-positives", type=int, default=2000)
    ap.add_argument("--ranking-positives", type=int, default=500)
    ap.add_argument("--num-neg-rank", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--no-baselines", action="store_true", help="Skip structural baselines (faster).")
    ap.add_argument("--no-ranking", action="store_true", help="Skip MRR/Hits@K (faster).")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    env = _env_info()
    print("Environment:", json.dumps(env, indent=2))
    print("\nLoading artifacts...")
    embeddings, data = ev.load_artifacts(root=args.root)

    scorers = {
        "mlp": ev.MLPScorer(embeddings, os.path.join(args.root, ev.DEFAULT_CHECKPOINT)),
        "cosine": ev.CosineScorer(embeddings),
    }
    if not args.no_baselines:
        print("Building global adjacency for structural baselines...")
        t0 = time.time()
        adj = ev._GlobalAdjacency(data)
        print(f"  adjacency built in {time.time()-t0:.1f}s ({adj.num_nodes:,} nodes)")
        scorers["common_neighbors"] = ev.CommonNeighborsScorer(adj)
        scorers["adamic_adar"] = ev.AdamicAdarScorer(adj)
        scorers["degree"] = ev.DegreeScorer(adj)
    scorers["random"] = ev.RandomScorer(seed=args.seed)
    scorer_names = list(scorers.keys())

    edge_types = ev.select_edge_types(data, args.min_edges, args.max_types)
    print(f"\nEvaluating {len(edge_types)} relation types with scorers: {scorer_names}\n")

    results = []
    for i, et in enumerate(edge_types, 1):
        t0 = time.time()
        r = ev.evaluate_edge_type(
            data, et, scorers, rng,
            sample_positives=args.sample_positives,
            ranking_positives=args.ranking_positives,
            num_neg_rank=args.num_neg_rank,
            n_boot=args.n_boot,
            compute_ranking=not args.no_ranking,
        )
        results.append(r)
        mlp_auc = r.scorers.get("mlp", {}).get("roc_auc", float("nan"))
        cos_auc = r.scorers.get("cosine", {}).get("roc_auc", float("nan"))
        print(f"[{i}/{len(edge_types)}] {'/'.join(et)}  n={r.num_positives_total}  "
              f"mlp_auc={mlp_auc:.3f} cos_auc={cos_auc:.3f}  ({time.time()-t0:.1f}s)")

    agg = _aggregate(results, scorer_names)

    # Calibration on pooled positives+negatives from learned scorers (largest relation)
    print("\nComputing calibration...")
    calib = {}
    big = max(results, key=lambda r: r.num_evaluated)
    st, _, dt = big.edge_type
    ei = data[big.edge_type].edge_index
    sel = rng.choice(ei.size(1), size=min(4000, ei.size(1)), replace=False)
    ps, pd = ei[0, sel].numpy(), ei[1, sel].numpy()
    existing = ev.existing_pairs_for_types(data, st, dt)
    ns, nd = ev.sample_filtered_negatives(existing, data[st].num_nodes, data[dt].num_nodes, len(ps), rng)
    y_true = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))]).astype(int)
    all_s = np.concatenate([ps, ns]); all_d = np.concatenate([pd, nd])
    for sn in ("cosine", "mlp"):
        y_score = scorers[sn].score(st, all_s, dt, all_d)
        calib[sn] = ev.calibrate_scores(y_true, y_score, method="isotonic", seed=args.seed)
        calib[sn]["calibrated_on"] = "/".join(big.edge_type)

    # Serialize
    out = {
        "config": vars(args),
        "environment": env,
        "scorers": scorer_names,
        "aggregate": agg,
        "calibration": calib,
        "per_edge_type": [
            {
                "edge_type": list(r.edge_type),
                "num_positives_total": r.num_positives_total,
                "num_evaluated": r.num_evaluated,
                "metrics": r.scorers,
            }
            for r in results
        ],
    }
    json_path = os.path.join(args.out_dir, "link_prediction_metrics.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {json_path}")

    md_path = os.path.join(args.out_dir, "link_prediction_summary.md")
    write_summary_md(md_path, vars(args), env, results, agg, scorer_names, calib)
    print(f"Wrote {md_path}")

    if not args.no_figures:
        figs = make_figures(results, agg, scorer_names, calib, args.out_dir)
        for k, v in figs.items():
            print(f"Wrote {v}")

    # Console headline
    print("\n" + "=" * 70)
    print("HEADLINE (macro mean over relations):")
    for sn in scorer_names:
        a = agg.get(sn, {})
        print(f"  {sn:16s} AUC={a.get('roc_auc_macro', float('nan')):.3f}  "
              f"AP={a.get('average_precision_macro', float('nan')):.3f}  "
              f"MRR={a.get('mrr_macro', float('nan')):.3f}  "
              f"Hits@10={a.get('hits@10_macro', float('nan')):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
