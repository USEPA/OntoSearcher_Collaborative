#!/usr/bin/env python3
"""
Quantitative link-prediction evaluation for the heterogeneous R-GCN embeddings.

This module evaluates the *frozen* node embeddings in ``nkb_rgcn_embeddings.pt``
(produced by ``src/gnn/proper_rgcn_hetero.py``) against the graph in
``improved_hetero_data.pt``. It reports the metrics reviewers asked for:
ROC-AUC, PR-AUC/Average Precision, Precision, Recall, F1, Accuracy, and the
ranking metrics MRR and Hits@K, each with bootstrap 95% confidence intervals.

Two scoring functions are evaluated so the paper can report both faithfully:

1. ``cosine``  - cosine similarity of the frozen embeddings rescaled to [0, 1].
   This is the score used by ``scripts/paper_prediction_report.py`` to produce
   the "high-confidence material-assay pairs" table.
2. ``mlp``     - the trained link-prediction head (an MLP) stored in
   ``best_rgcn_model.pt`` applied to the concatenated embeddings. This is the
   objective the R-GCN was actually trained on (source of the reported AUC).

Conventional (non-learned) baselines are evaluated under the *identical*
protocol so the embedding model's lift over plain graph queries is quantified:
common-neighbors, Adamic-Adar, destination-degree (popularity), and random.

IMPORTANT (honesty / limitations):
    Knowledge-graph embeddings here are *transductive*: the embeddings were
    trained over the full graph, so held-out test edges were visible during
    representation learning (this is standard for ComplEx/TransE-style KGE).
    Absolute numbers should be read in that light; the structural baselines are
    run under the same protocol so relative comparisons remain fair.

All randomness is seeded for reproducibility.

Run via ``scripts/evaluate_link_prediction.py`` (recommended) or import the
functions here directly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required. Use the 'graph_env' conda env: "
        "/Users/pranavsingh/miniforge3/envs/graph_env/bin/python"
    ) from exc

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #

DEFAULT_EMBEDDINGS = "nkb_rgcn_embeddings.pt"
DEFAULT_HETERO = "improved_hetero_data.pt"
DEFAULT_CHECKPOINT = "best_rgcn_model.pt"


def load_artifacts(
    root: str = ".",
    embeddings_path: str = DEFAULT_EMBEDDINGS,
    hetero_path: str = DEFAULT_HETERO,
):
    """Load frozen embeddings dict and the HeteroData graph (CPU)."""
    torch.set_num_threads(int(os.environ.get("EVAL_NUM_THREADS", "4")))
    emb_p = os.path.join(root, embeddings_path)
    het_p = os.path.join(root, hetero_path)
    if not os.path.isfile(emb_p):
        raise FileNotFoundError(f"Embeddings not found: {emb_p}")
    if not os.path.isfile(het_p):
        raise FileNotFoundError(f"HeteroData not found: {het_p}")
    embeddings = torch.load(emb_p, weights_only=False, map_location="cpu")
    data = torch.load(het_p, weights_only=False, map_location="cpu")
    return embeddings, data


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #


class Scorer:
    """Scores (src_type, src_idx[], dst_type, dst_idx[]) -> np.ndarray in some range."""

    name = "base"
    higher_is_better = True
    # Ranking (MRR/Hits@K) corrupts each positive with many candidate tails and
    # re-scores them. This is cheap for vectorized scorers but O(candidates) set
    # intersections for structural baselines, so it is disabled for those.
    cheap_ranking = True

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:  # noqa: D401
        raise NotImplementedError


class CosineScorer(Scorer):
    """Cosine similarity rescaled to [0, 1] (matches paper_prediction_report.py)."""

    name = "cosine"

    def __init__(self, embeddings: Dict[str, "torch.Tensor"]):
        # Pre-normalize once for speed.
        self.norm = {k: F.normalize(v, p=2, dim=1) for k, v in embeddings.items()}

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        a = self.norm[src_type][torch.as_tensor(src_idx, dtype=torch.long)]
        b = self.norm[dst_type][torch.as_tensor(dst_idx, dtype=torch.long)]
        cos = (a * b).sum(dim=1)
        return ((cos + 1.0) / 2.0).clamp_(0.0, 1.0).numpy()


class MLPScorer(Scorer):
    """The trained link-prediction MLP head applied to frozen embeddings.

    Rebuilds the exact ``link_predictor`` Sequential from proper_rgcn_hetero.py
    (Linear(2h, h) -> ReLU -> Dropout -> Linear(h, h//2) -> ReLU -> Dropout ->
    Linear(h//2, 1) -> Sigmoid) and loads its weights from ``best_rgcn_model.pt``.
    No RGCN message passing is needed because the input embeddings are the model
    forward output already saved to disk.
    """

    name = "mlp"

    def __init__(self, embeddings, checkpoint_path: str):
        self.embeddings = embeddings
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        w0 = state["link_predictor.0.weight"]  # (h, 2h)
        hidden = w0.size(0)
        two_h = w0.size(1)
        h_half = state["link_predictor.3.weight"].size(0)
        self.mlp = nn.Sequential(
            nn.Linear(two_h, hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, h_half),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(h_half, 1),
            nn.Sigmoid(),
        )
        prefix = "link_predictor."
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.mlp.load_state_dict(sub)
        self.mlp.eval()

    @torch.no_grad()
    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        a = self.embeddings[src_type][torch.as_tensor(src_idx, dtype=torch.long)]
        b = self.embeddings[dst_type][torch.as_tensor(dst_idx, dtype=torch.long)]
        x = torch.cat([a, b], dim=-1)
        return self.mlp(x).squeeze(-1).numpy()


# ---- Structural / "conventional query" baselines --------------------------- #


class _GlobalAdjacency:
    """Undirected sparse adjacency over the global (offset) node index space.

    Used by the structural baselines. Nodes of each type occupy a contiguous
    block; ``offset[node_type]`` gives the start index.
    """

    def __init__(self, data):
        from scipy import sparse

        self.offset: Dict[str, int] = {}
        cur = 0
        for nt in data.node_types:
            self.offset[nt] = cur
            cur += data[nt].num_nodes
        self.num_nodes = cur

        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        for et in data.edge_types:
            st, _, dt = et
            ei = data[et].edge_index
            if ei.numel() == 0:
                continue
            s = ei[0].numpy() + self.offset[st]
            d = ei[1].numpy() + self.offset[dt]
            rows.append(s)
            cols.append(d)
        r = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        c = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        # symmetrize
        ri = np.concatenate([r, c])
        ci = np.concatenate([c, r])
        vals = np.ones(ri.shape[0], dtype=np.float32)
        A = sparse.csr_matrix((vals, (ri, ci)), shape=(self.num_nodes, self.num_nodes))
        A.data[:] = 1.0  # binary adjacency
        self.A = A
        self.degree = np.asarray(A.sum(axis=1)).ravel()
        # 1/log(deg) weights for Adamic-Adar (guard deg<=1)
        with np.errstate(divide="ignore"):
            self.inv_log_deg = np.where(self.degree > 1, 1.0 / np.log(self.degree), 0.0)

    def gidx(self, node_type: str, idx) -> np.ndarray:
        return np.asarray(idx, dtype=np.int64) + self.offset[node_type]


class CommonNeighborsScorer(Scorer):
    name = "common_neighbors"
    cheap_ranking = False

    def __init__(self, adj: _GlobalAdjacency):
        self.adj = adj

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        A = self.adj.A
        gu = self.adj.gidx(src_type, src_idx)
        gv = self.adj.gidx(dst_type, dst_idx)
        out = np.empty(len(gu), dtype=np.float32)
        for i in range(len(gu)):
            ru = A.getrow(gu[i]).indices
            rv = A.getrow(gv[i]).indices
            out[i] = np.intersect1d(ru, rv, assume_unique=True).size
        return out


class AdamicAdarScorer(Scorer):
    name = "adamic_adar"
    cheap_ranking = False

    def __init__(self, adj: _GlobalAdjacency):
        self.adj = adj

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        A = self.adj.A
        w = self.adj.inv_log_deg
        gu = self.adj.gidx(src_type, src_idx)
        gv = self.adj.gidx(dst_type, dst_idx)
        out = np.empty(len(gu), dtype=np.float32)
        for i in range(len(gu)):
            ru = A.getrow(gu[i]).indices
            rv = A.getrow(gv[i]).indices
            common = np.intersect1d(ru, rv, assume_unique=True)
            out[i] = float(w[common].sum()) if common.size else 0.0
        return out


class DegreeScorer(Scorer):
    """Popularity baseline: score = degree of the destination node."""

    name = "degree"

    def __init__(self, adj: _GlobalAdjacency):
        self.adj = adj

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        gv = self.adj.gidx(dst_type, dst_idx)
        return self.adj.degree[gv].astype(np.float32)


class RandomScorer(Scorer):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, src_type, src_idx, dst_type, dst_idx) -> np.ndarray:
        return self.rng.random(len(np.asarray(src_idx)))


# --------------------------------------------------------------------------- #
# Negative sampling (filtered) & existing-edge index
# --------------------------------------------------------------------------- #


def existing_pairs_for_types(data, src_type: str, dst_type: str) -> set:
    """All (src_idx, dst_idx) pairs linking src_type->dst_type under ANY relation.

    Used to *filter* negatives so a sampled negative is never a real edge.
    """
    pairs = set()
    for et in data.edge_types:
        if et[0] == src_type and et[-1] == dst_type:
            ei = data[et].edge_index
            for k in range(ei.size(1)):
                pairs.add((int(ei[0, k]), int(ei[1, k])))
    return pairs


def sample_filtered_negatives(
    existing: set,
    n_src: int,
    n_dst: int,
    num: int,
    rng: np.random.Generator,
    max_tries_factor: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``num`` (src, dst) pairs not present in ``existing``."""
    src_out = np.empty(num, dtype=np.int64)
    dst_out = np.empty(num, dtype=np.int64)
    got = 0
    tries = 0
    max_tries = num * max_tries_factor
    while got < num and tries < max_tries:
        s = int(rng.integers(0, n_src))
        d = int(rng.integers(0, n_dst))
        tries += 1
        if (s, d) in existing:
            continue
        src_out[got] = s
        dst_out[got] = d
        got += 1
    if got < num:  # graph nearly complete for this type; return what we have
        src_out = src_out[:got]
        dst_out = dst_out[:got]
    return src_out, dst_out


# --------------------------------------------------------------------------- #
# Metrics + bootstrap CIs
# --------------------------------------------------------------------------- #


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
    }
    return out


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: int = 101) -> Tuple[float, float]:
    """Return (threshold, f1) maximizing F1 over a score-quantile grid."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if hi <= lo:
        return 0.5, f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
    best_t, best = lo, -1.0
    for t in np.linspace(lo, hi, grid):
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t, float(best)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via stratified bootstrap resampling."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    point = metric_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float(point), float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ps = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ns = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ps, ns])
        try:
            stats[b] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            stats[b] = np.nan
    lo = float(np.nanpercentile(stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---- Ranking metrics (MRR, Hits@K) ----------------------------------------- #


def ranking_metrics(
    scorer: Scorer,
    src_type: str,
    dst_type: str,
    pos_src: np.ndarray,
    pos_dst: np.ndarray,
    existing: set,
    n_dst: int,
    rng: np.random.Generator,
    num_neg: int = 100,
    ks: Sequence[int] = (1, 3, 10),
) -> Dict[str, float]:
    """Filtered tail-corruption ranking: for each positive (s, d+), rank d+ against
    ``num_neg`` sampled corrupt tails (excluding known edges). Returns MRR + Hits@K.
    """
    reciprocal = []
    hits = {k: 0 for k in ks}
    n = len(pos_src)
    for i in range(n):
        s = int(pos_src[i])
        d_true = int(pos_dst[i])
        # sample corrupt tails
        negs = set()
        tries = 0
        while len(negs) < num_neg and tries < num_neg * 50:
            cand = int(rng.integers(0, n_dst))
            tries += 1
            if cand == d_true or (s, cand) in existing:
                continue
            negs.add(cand)
        cand_dst = np.array([d_true] + list(negs), dtype=np.int64)
        cand_src = np.full(len(cand_dst), s, dtype=np.int64)
        scores = scorer.score(src_type, cand_src, dst_type, cand_dst)
        if not scorer.higher_is_better:
            scores = -scores
        # rank of the true tail (index 0); 1-based, ties broken pessimistically
        true_score = scores[0]
        rank = 1 + int(np.sum(scores[1:] > true_score))
        # random tie-break among equals
        ties = int(np.sum(scores[1:] == true_score))
        if ties:
            rank += int(rng.integers(0, ties + 1))
        reciprocal.append(1.0 / rank)
        for k in ks:
            if rank <= k:
                hits[k] += 1
    out = {"mrr": float(np.mean(reciprocal)) if reciprocal else float("nan")}
    for k in ks:
        out[f"hits@{k}"] = float(hits[k] / n) if n else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Per-edge-type evaluation
# --------------------------------------------------------------------------- #


@dataclass
class EdgeTypeResult:
    edge_type: Tuple[str, str, str]
    num_positives_total: int
    num_evaluated: int
    scorers: Dict[str, dict] = field(default_factory=dict)


def select_edge_types(data, min_edges: int, max_types: int) -> List[Tuple[str, str, str]]:
    counts = [(et, data[et].edge_index.size(1)) for et in data.edge_types]
    counts = [(et, c) for et, c in counts if c >= min_edges]
    counts.sort(key=lambda x: -x[1])
    return [et for et, _ in counts[:max_types]]


def evaluate_edge_type(
    data,
    edge_type: Tuple[str, str, str],
    scorers: Dict[str, Scorer],
    rng: np.random.Generator,
    sample_positives: int = 2000,
    ranking_positives: int = 1000,
    num_neg_rank: int = 100,
    n_boot: int = 1000,
    compute_ranking: bool = True,
) -> EdgeTypeResult:
    st, _, dt = edge_type
    ei = data[edge_type].edge_index
    total = ei.size(1)
    n_src = data[st].num_nodes
    n_dst = data[dt].num_nodes

    # sample positives
    if total > sample_positives:
        sel = rng.choice(total, size=sample_positives, replace=False)
    else:
        sel = np.arange(total)
    pos_src = ei[0, sel].numpy()
    pos_dst = ei[1, sel].numpy()
    n_pos = len(pos_src)

    existing = existing_pairs_for_types(data, st, dt)
    neg_src, neg_dst = sample_filtered_negatives(existing, n_src, n_dst, n_pos, rng)
    n_neg = len(neg_src)

    all_src = np.concatenate([pos_src, neg_src])
    all_dst = np.concatenate([pos_dst, neg_dst])
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)

    res = EdgeTypeResult(edge_type=edge_type, num_positives_total=total, num_evaluated=n_pos)

    # positives for ranking (subsample)
    if compute_ranking:
        rp = min(ranking_positives, n_pos)
        ridx = rng.choice(n_pos, size=rp, replace=False)
        rank_src = pos_src[ridx]
        rank_dst = pos_dst[ridx]

    for sname, scorer in scorers.items():
        y_score = scorer.score(st, all_src, dt, all_dst)
        if not scorer.higher_is_better:
            y_score = -y_score
        # for calibration/threshold metrics we want scores comparable; min-max for non-prob scorers
        prob_like = sname in ("cosine", "mlp")
        thr = 0.5 if prob_like else _mid_threshold(y_score)
        m = binary_metrics(y_true, y_score, thr)
        bt, bf1 = best_f1_threshold(y_true, y_score)
        m["best_f1"] = bf1
        m["best_f1_threshold"] = bt
        # bootstrap CIs for the headline metrics
        _, auc_lo, auc_hi = bootstrap_ci(y_true, y_score, lambda a, b: roc_auc_score(a, b) if len(set(a.tolist())) > 1 else float("nan"), n_boot=n_boot)
        _, ap_lo, ap_hi = bootstrap_ci(y_true, y_score, lambda a, b: average_precision_score(a, b) if a.sum() > 0 else float("nan"), n_boot=n_boot)
        m["roc_auc_ci95"] = [auc_lo, auc_hi]
        m["average_precision_ci95"] = [ap_lo, ap_hi]
        # score distributions
        m["pos_score_mean"] = float(np.mean(y_score[:n_pos]))
        m["neg_score_mean"] = float(np.mean(y_score[n_pos:]))
        # ranking (skipped for scorers where it is prohibitively expensive)
        if compute_ranking and getattr(scorer, "cheap_ranking", True):
            rm = ranking_metrics(
                scorer, st, dt, rank_src, rank_dst, existing, n_dst, rng,
                num_neg=num_neg_rank,
            )
            m.update(rm)
        res.scorers[sname] = m

    return res


def _mid_threshold(scores: np.ndarray) -> float:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    return (lo + hi) / 2.0 if hi > lo else 0.5


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibrate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "isotonic",
    seed: int = 42,
    test_frac: float = 0.5,
) -> Dict[str, object]:
    """Fit a calibrator on a train split and report Brier score + reliability
    curve on the held-out split.

    Answers the "similarity score is not a calibrated probability" critique by
    turning raw scores into calibrated probabilities and quantifying quality.
    Returns a dict with pre/post Brier scores and reliability-curve arrays.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_true))
    n_test = int(len(idx) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    ytr, str_ = y_true[train_idx], y_score[train_idx]
    yte, ste = y_true[test_idx], y_score[test_idx]

    # scale raw scores into [0,1] for a fair pre-calibration Brier
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    scale = lambda s: (s - lo) / (hi - lo) if hi > lo else np.clip(s, 0, 1)

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(str_, ytr)
        prob_te = cal.predict(ste)
    else:  # platt / sigmoid
        cal = LogisticRegression()
        cal.fit(str_.reshape(-1, 1), ytr)
        prob_te = cal.predict_proba(ste.reshape(-1, 1))[:, 1]

    prob_te = np.clip(prob_te, 0.0, 1.0)
    brier_pre = float(brier_score_loss(yte, np.clip(scale(ste), 0, 1))) if len(set(yte.tolist())) > 1 else float("nan")
    brier_post = float(brier_score_loss(yte, prob_te)) if len(set(yte.tolist())) > 1 else float("nan")
    try:
        frac_pos, mean_pred = calibration_curve(yte, prob_te, n_bins=10, strategy="quantile")
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])
    return {
        "method": method,
        "brier_pre": brier_pre,
        "brier_post": brier_post,
        "reliability_mean_predicted": mean_pred.tolist(),
        "reliability_fraction_positive": frac_pos.tolist(),
        "n_test": int(len(test_idx)),
    }
