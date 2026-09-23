#!/usr/bin/env python3
"""
ComplEx knowledge-graph-embedding baseline evaluation (no pykeen required).

Provides a second, independent link-prediction model to benchmark the R-GCN
against, reporting the ranking metrics reviewers like (MRR, Hits@K) plus
ROC-AUC / PR-AUC / Precision / Recall / F1 with 95% bootstrap CIs.

It loads the pre-trained ComplEx artifacts in results_nkb/ (produced earlier via
PyKEEN) directly as tensors + id maps, so it does NOT need pykeen installed:

    results_nkb/complex_nkb_entity_to_id.pt      URI -> entity id (dict)
    results_nkb/complex_nkb_relation_to_id.pt    URI -> relation id (dict)
    results_nkb/complex_nkb_entity_embeddings.pt   complex64 [n_ent, dim]
    results_nkb/complex_nkb_relation_embeddings.pt complex64 [n_rel, dim]

ComplEx score(h, r, t) = Re( sum_k  e_h[k] * e_r[k] * conj(e_t[k]) ).

Positive triples are reconstructed from the NKB RDF (URI-URI triples whose
subject, predicate, object are all present in the id maps), matching how the
PyKEEN TriplesFactory was originally built. The extracted triples are cached to
results_nkb/complex_nkb_triples_cache.pt.

Run with the working env:
    /Users/pranavsingh/miniforge3/envs/graph_env/bin/python \
        scripts/evaluate_complex_baseline.py

NOTE: This is a transductive KGE evaluation (as is standard for ComplEx). Use it
alongside scripts/evaluate_link_prediction.py for a model-to-model comparison.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "gnn"))

import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import evaluation as ev  # reuse bootstrap_ci  # noqa: E402

RESULTS = os.path.join(REPO_ROOT, "results_nkb")
CACHE = os.path.join(RESULTS, "complex_nkb_triples_cache.pt")
DEFAULT_RDF = os.path.join(REPO_ROOT, "mappings", "NKB_RDF_V3.ttl")


def load_complex():
    e2id = torch.load(os.path.join(RESULTS, "complex_nkb_entity_to_id.pt"), weights_only=False, map_location="cpu")
    r2id = torch.load(os.path.join(RESULTS, "complex_nkb_relation_to_id.pt"), weights_only=False, map_location="cpu")
    ent = torch.load(os.path.join(RESULTS, "complex_nkb_entity_embeddings.pt"), weights_only=False, map_location="cpu")
    rel = torch.load(os.path.join(RESULTS, "complex_nkb_relation_embeddings.pt"), weights_only=False, map_location="cpu")
    return e2id, r2id, ent, rel


def extract_triples(e2id, r2id, rdf_path):
    """Reconstruct integer triples (h, r, t) from the RDF using the id maps."""
    if os.path.isfile(CACHE):
        print(f"Loading cached triples: {CACHE}")
        return torch.load(CACHE, weights_only=False)
    print(f"Parsing RDF to reconstruct ComplEx triples: {rdf_path}")
    print("(one-time; cached afterwards)")
    import rdflib

    g = rdflib.Graph()
    t0 = time.time()
    g.parse(rdf_path, format="turtle")
    print(f"  parsed {len(g):,} RDF triples in {time.time()-t0:.1f}s")
    triples = []
    for s, p, o in g:
        su, pu, ou = str(s), str(p), str(o)
        hi = e2id.get(su)
        ri = r2id.get(pu)
        ti = e2id.get(ou)
        if hi is not None and ri is not None and ti is not None:
            triples.append((hi, ri, ti))
    arr = np.asarray(triples, dtype=np.int64)
    print(f"  matched {len(arr):,} URI-URI triples covering the id maps")
    torch.save(arr, CACHE)
    return arr


def complex_score(ent, rel, h, r, t):
    """Vectorized ComplEx score for arrays of (h, r, t) integer ids."""
    eh = ent[torch.as_tensor(h, dtype=torch.long)]
    er = rel[torch.as_tensor(r, dtype=torch.long)]
    et = ent[torch.as_tensor(t, dtype=torch.long)]
    return (eh * er * et.conj()).sum(dim=1).real.numpy()


def binary_block(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rdf", default=DEFAULT_RDF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-positives", type=int, default=3000, help="Positives sampled for AUC/AP/P/R/F1.")
    ap.add_argument("--ranking-positives", type=int, default=1000)
    ap.add_argument("--num-neg-rank", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "evaluation"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    e2id, r2id, ent, rel = load_complex()
    n_ent = ent.size(0)
    print(f"ComplEx: {n_ent:,} entities, {rel.size(0)} relations, dim={ent.size(1)} (complex)")
    id2rel = {v: k for k, v in r2id.items()}

    triples = extract_triples(e2id, r2id, args.rdf)
    if len(triples) == 0:
        print("ERROR: no triples reconstructed. Check RDF path / id maps.")
        sys.exit(1)

    known = set(map(tuple, triples.tolist()))  # for filtered negatives

    # group triples by relation
    rels = np.unique(triples[:, 1])
    per_rel = {int(r): triples[triples[:, 1] == r] for r in rels}

    per_rel_out = []
    overall_true, overall_score = [], []
    macro = {k: [] for k in ["roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "mrr", "hits@1", "hits@3", "hits@10"]}

    for ri in sorted(per_rel.keys(), key=lambda r: -len(per_rel[r])):
        pos = per_rel[ri]
        npos_total = len(pos)
        if npos_total < 20:
            continue
        sel = rng.choice(npos_total, size=min(args.sample_positives, npos_total), replace=False)
        ph, pr, pt = pos[sel, 0], pos[sel, 1], pos[sel, 2]
        n = len(ph)
        # filtered negatives via tail corruption
        nh, nr, nt = ph.copy(), pr.copy(), np.empty(n, dtype=np.int64)
        for i in range(n):
            while True:
                cand = int(rng.integers(0, n_ent))
                if (int(nh[i]), int(nr[i]), cand) not in known:
                    nt[i] = cand
                    break
        y_true = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
        all_h = np.concatenate([ph, nh]); all_r = np.concatenate([pr, nr]); all_t = np.concatenate([pt, nt])
        y_score = complex_score(ent, rel, all_h, all_r, all_t)
        thr = float(np.median(y_score))
        m = binary_block(y_true, y_score, thr)
        _, alo, ahi = ev.bootstrap_ci(y_true, y_score, lambda a, b: roc_auc_score(a, b), n_boot=args.n_boot)
        m["roc_auc_ci95"] = [alo, ahi]

        # ranking: tail corruption with num_neg candidates
        rp = min(args.ranking_positives, n)
        ridx = rng.choice(n, size=rp, replace=False)
        recip, hits = [], {1: 0, 3: 0, 10: 0}
        for i in ridx:
            h_, r_, t_ = int(ph[i]), int(pr[i]), int(pt[i])
            negs = set()
            while len(negs) < args.num_neg_rank:
                c = int(rng.integers(0, n_ent))
                if c != t_ and (h_, r_, c) not in known:
                    negs.add(c)
            cand_t = np.array([t_] + list(negs), dtype=np.int64)
            sc = complex_score(ent, rel, np.full(len(cand_t), h_), np.full(len(cand_t), r_), cand_t)
            rank = 1 + int(np.sum(sc[1:] > sc[0]))
            ties = int(np.sum(sc[1:] == sc[0]))
            if ties:
                rank += int(rng.integers(0, ties + 1))
            recip.append(1.0 / rank)
            for k in hits:
                if rank <= k:
                    hits[k] += 1
        m["mrr"] = float(np.mean(recip))
        for k in hits:
            m[f"hits@{k}"] = float(hits[k] / rp)

        for k in macro:
            if k in m and not (isinstance(m[k], float) and np.isnan(m[k])):
                macro[k].append(m[k])
        overall_true.append(y_true); overall_score.append(y_score)

        per_rel_out.append({
            "relation_uri": id2rel.get(ri, str(ri)),
            "relation_id": ri,
            "num_positives_total": npos_total,
            "num_evaluated": n,
            "metrics": m,
        })
        print(f"  rel {ri:2d} ({id2rel.get(ri,'?').split('/')[-1].split('#')[-1]:<20}) "
              f"n={npos_total:>7}  AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}  "
              f"MRR={m['mrr']:.3f}  H@10={m['hits@10']:.3f}")

    agg = {f"{k}_macro": float(np.mean(v)) for k, v in macro.items() if v}
    ot = np.concatenate(overall_true); os_ = np.concatenate(overall_score)
    agg["roc_auc_micro"] = float(roc_auc_score(ot, os_))
    agg["average_precision_micro"] = float(average_precision_score(ot, os_))

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "executable": sys.executable,
    }
    out = {"config": vars(args), "environment": env, "model": "ComplEx",
           "aggregate": agg, "per_relation": per_rel_out}
    jp = os.path.join(args.out_dir, "complex_baseline_metrics.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {jp}")
    print("=" * 60)
    print("ComplEx BASELINE (macro over relations):")
    print(f"  AUC={agg.get('roc_auc_macro', float('nan')):.3f}  "
          f"AP={agg.get('average_precision_macro', float('nan')):.3f}  "
          f"MRR={agg.get('mrr_macro', float('nan')):.3f}  "
          f"Hits@10={agg.get('hits@10_macro', float('nan')):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
