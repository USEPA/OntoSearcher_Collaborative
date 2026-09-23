#!/usr/bin/env python3
"""
Generate numbers and tables for the paper Prediction section.

Run from repo root:
  python scripts/paper_prediction_report.py

Files loaded when scoring (no --csv):
  - nkb_rgcn_embeddings.pt  (node embeddings; from proper_rgcn_hetero.py)
  - improved_hetero_data.pt  (PyG HeteroData; from improved_rdf_hetero_converter.py)
These are the correct artifacts; no separate "model" file is loaded.

If you get a segmentation fault when loading torch (common on some Macs with PyTorch/PyG),
see docs/SETUP_PAPER_PREDICTION.md for steps to get PyTorch running (conda, torch 2.2.2, CPU).
Alternatively generate the CSV on a machine where PyTorch runs (e.g. cluster, Colab), then run:
  python scripts/paper_prediction_report.py --csv high_confidence_material_assay_pairs.csv
This skips torch entirely and only writes the summary from the existing CSV.

Outputs:
  - high_confidence_material_assay_pairs.csv  (unless --csv given); default 156 pairs, ranked by score, sampled across materials
  - prediction_section_summary.txt
"""

import argparse
import os
import random
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_PATH = os.path.join(REPO_ROOT, "nkb_rgcn_embeddings.pt")
HETERO_DATA_PATH = os.path.join(REPO_ROOT, "improved_hetero_data.pt")
NODE_URIS_PATH = os.path.join(REPO_ROOT, "improved_hetero_data_node_uris.pkl")
NETWORKX_GRAPH_PATH = os.path.join(REPO_ROOT, "networkx_graph.pkl")
OUTPUT_DIR = REPO_ROOT
MAPPINGS_DIR = os.path.join(REPO_ROOT, "mappings")
DEFAULT_RDF_PATH = os.path.join(REPO_ROOT, "mappings", "NKB_RDF_V3.ttl")


def _find_rdf_in_mappings():
    """Use default NKB .ttl if present; otherwise first .ttl in mappings/."""
    if os.path.isfile(DEFAULT_RDF_PATH):
        return DEFAULT_RDF_PATH
    if not os.path.isdir(MAPPINGS_DIR):
        return None
    for name in sorted(os.listdir(MAPPINGS_DIR)):
        if name.endswith(".ttl"):
            return os.path.join(MAPPINGS_DIR, name)
    return None


def get_existing_material_assay_edges(data):
    """Return set of (material_idx, assay_idx) that exist in the graph.
    Edge types in HeteroData are (src_type, relation, dst_type) tuples.
    Matches node types case-insensitively.
    """
    existing = set()
    for edge_type in list(data.edge_types):
        try:
            st = edge_type[0] if hasattr(edge_type, "__getitem__") else None
            dt = edge_type[-1] if hasattr(edge_type, "__getitem__") and len(edge_type) >= 2 else None
            if st is None or dt is None:
                continue
            st_str = str(st).lower() if st is not None else ""
            dt_str = str(dt).lower() if dt is not None else ""
            if st_str != "material" or dt_str != "assay":
                continue
            edge_index = data[edge_type].edge_index
            for c in range(edge_index.size(1)):
                src = edge_index[0, c].item()
                dst = edge_index[1, c].item()
                existing.add((int(src), int(dst)))
        except (KeyError, TypeError, IndexError, AttributeError):
            continue
    return existing


def _load_node_uris():
    """Load node_type -> [uri0, uri1, ...] from converter output. Returns None if missing."""
    if not os.path.isfile(NODE_URIS_PATH):
        return None
    try:
        import pickle
        with open(NODE_URIS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_node_uris_from_networkx(nx_path):
    """
    Build material/assay URI lists from networkx_graph.pkl in the same order the improved
    converter uses (iterate graph.nodes(), filter by entity_type). Use when
    improved_hetero_data_node_uris.pkl is not available.
    Returns {'material': [uri, ...], 'assay': [uri, ...]} or None.
    """
    if not nx_path or not os.path.isfile(nx_path):
        return None
    try:
        import pickle
        with open(nx_path, "rb") as f:
            g = pickle.load(f)
    except Exception:
        return None
    entity_types = {"material", "assay", "result", "parameters", "additive", "medium", "publication", "contam", "materialfg", "molecularresult"}
    by_type = {"material": [], "assay": []}
    for node in g.nodes():
        node_str = str(node)
        if not node_str.startswith("http://example.org/"):
            continue
        et = g.nodes[node].get("entity_type", "unknown")
        if et in entity_types and et in by_type:
            by_type[et].append(str(node))
    if not by_type["material"] and not by_type["assay"]:
        return None
    return by_type


def _load_uri_to_label(rdf_path, uris_to_fetch):
    """Load RDF and extract a human-readable label for each URI.
    Tries rdfs:label first; for NKB example.org entities (often no label) uses first
    string literal that looks like a name (e.g. ncit:C25372 'physical characterization').
    """
    if not uris_to_fetch or not rdf_path or not os.path.isfile(rdf_path):
        return {}
    try:
        import rdflib
        from rdflib import RDFS
        from rdflib import Literal
        g = rdflib.Graph()
        g.parse(rdf_path, format="turtle")
        uri_to_label = {}
        for uri in uris_to_fetch:
            uri = str(uri).strip()
            if not uri:
                continue
            try:
                node = rdflib.URIRef(uri) if not uri.startswith("_:") else rdflib.BNode(uri[2:])
            except Exception:
                continue
            label = g.value(node, RDFS.label)
            if label is not None:
                uri_to_label[uri] = str(label)
            else:
                for pred, obj in g.predicate_objects(node):
                    if "label" in str(pred).lower():
                        uri_to_label[uri] = str(obj)
                        break
            if uri not in uri_to_label and "example.org" in uri:
                for pred, obj in g.predicate_objects(node):
                    if isinstance(obj, Literal) and obj.datatype is None:
                        s = str(obj).strip()
                        if len(s) > 3 and not s.replace(".", "").replace("-", "").isdigit() and "10." not in s[:5]:
                            uri_to_label[uri] = s
                            break
            if uri not in uri_to_label:
                uri_to_label[uri] = (uri.split("/")[-1].split("#")[-1] or uri)
        return uri_to_label
    except Exception:
        return {}


def run_material_assay_predictions(threshold=0.98, max_pairs_cap=200_000, batch_size=2000, top_k=156, diversify_across_materials=True):
    """
    Score material-assay pairs using only embeddings (no RDF, no full predictor).
    Probability = (cosine_similarity + 1) / 2 so scores are in [0,1]; threshold 0.98 means cos_sim >= 0.96.
    Returns at most top_k pairs above threshold, ranked by score descending.
    If diversify_across_materials is True, caps pairs per material so the output is spread across many materials.
    Returns DataFrame: material_idx, assay_idx, probability, rank (1-based).
    """
    # Import torch first (before numpy/pandas) to reduce segfault risk on Mac ARM
    import torch
    import torch.nn.functional as F

    # Mac/ARM workarounds: force CPU, limit threads, avoid MPS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    torch.set_num_threads(4)
    device = torch.device("cpu")

    print("Loading embeddings and graph (torch only)...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"ERROR: {EMBEDDINGS_PATH} not found.")
        return __import__("pandas").DataFrame()
    if not os.path.exists(HETERO_DATA_PATH):
        print(f"ERROR: {HETERO_DATA_PATH} not found.")
        return __import__("pandas").DataFrame()

    embeddings = torch.load(EMBEDDINGS_PATH, weights_only=False, map_location=device)
    data = torch.load(HETERO_DATA_PATH, weights_only=False, map_location=device)

    if "material" not in embeddings or "assay" not in embeddings:
        print("ERROR: 'material' or 'assay' not in embeddings. Check embedding file.")
        return __import__("pandas").DataFrame()

    mat_emb = embeddings["material"]  # [N_material, dim]
    assay_emb = embeddings["assay"]   # [N_assay, dim]
    n_material, n_assay = mat_emb.size(0), assay_emb.size(0)
    existing = get_existing_material_assay_edges(data)
    # Report edge-type situation (NKB improved_hetero_data often has entity→concept, not entity→entity)
    mat_assay_types = [et for et in list(data.edge_types) if hasattr(et, "__getitem__") and len(et) >= 2 and str(et[0]).lower() == "material" and str(et[-1]).lower() == "assay"]
    if existing:
        print(f"Materials: {n_material}, Assays: {n_assay}, Existing edges: {len(existing)}")
    else:
        if mat_assay_types:
            print(f"Note: edge types {mat_assay_types} found but no edges loaded (check keys).")
        else:
            print("Note: graph has no direct (material, *, assay) edges (NKB links entities to concepts/blank nodes).")
            print("      All material-assay pairs are treated as non-existing; predictions use embedding similarity only.")
        print(f"Materials: {n_material}, Assays: {n_assay}, Existing edges: {len(existing)}")

    # Collect candidate (material, assay) pairs not in graph — sample diversely across materials and assays
    random.seed(42)
    candidates_set = set()
    while len(candidates_set) < max_pairs_cap:
        i = random.randint(0, n_material - 1) if n_material else 0
        j = random.randint(0, n_assay - 1) if n_assay else 0
        if (i, j) not in existing:
            candidates_set.add((i, j))
    candidates = list(candidates_set)
    print(f"Scoring {len(candidates)} non-existing pairs (diverse sample of materials and assays)...")

    # Score: cosine similarity in [-1,1] -> map to [0,1] so threshold is achievable.
    scores = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        for (i, j) in batch:
            a = mat_emb[i : i + 1]
            b = assay_emb[j : j + 1]
            sim = F.cosine_similarity(a, b).item()
            p = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            if p >= threshold:
                scores.append((i, j, p))
        if (start // batch_size) % 50 == 0 and start > 0:
            print(f"   Processed {start + len(batch)}, above threshold so far: {len(scores)}")

    # Keep top_k by score; if diversifying, cap pairs per material so output spans many materials
    scores.sort(key=lambda x: x[2], reverse=True)
    if diversify_across_materials and top_k > 0 and scores:
        # Max pairs per material so we get ~top_k pairs spread across as many materials as possible
        n_materials_approx = len(set(i for i, _, _ in scores))
        max_per_material = max(1, (top_k + n_materials_approx - 1) // n_materials_approx)
        material_count = {}
        selected = []
        for (i, j, p) in scores:
            if len(selected) >= top_k:
                break
            count_so_far = material_count.get(i, 0)
            if count_so_far < max_per_material:
                material_count[i] = count_so_far + 1
                selected.append((i, j, p))
        scores = selected
    else:
        scores = scores[:top_k]
    print(f"Found {len(scores)} pairs with score >= {threshold} (output capped at top {top_k}, ranked by score)")
    if diversify_across_materials and scores:
        n_mat = len(set(i for i, _, _ in scores))
        print(f"Sampled across {n_mat} materials (max {max(1, (top_k + n_mat - 1) // n_mat)} pairs per material).")
    print("Note: Score is embedding similarity (cosine rescaled to [0,1]), not a calibrated link probability.")

    rows = [{"material_idx": i, "assay_idx": j, "probability": p, "similarity_score": p, "rank": r}
            for r, (i, j, p) in enumerate(scores, start=1)]
    return __import__("pandas").DataFrame(rows)


def add_uris_and_labels(df, rdf_path=None, networkx_path=None):
    """
    Add material_uri, assay_uri, material_label, assay_label columns.
    Uses improved_hetero_data_node_uris.pkl if present; else falls back to networkx_graph.pkl
    (same node order as improved converter: graph iteration + entity_type filter).
    """
    if df is None or df.empty:
        return df
    node_uris = _load_node_uris()
    source = "pkl"
    if node_uris is None:
        node_uris = _load_node_uris_from_networkx(networkx_path or NETWORKX_GRAPH_PATH)
        source = "networkx"
    if node_uris is None or ("material" not in node_uris and "assay" not in node_uris):
        return df
    mat_uris = node_uris.get("material", [])
    assay_uris = node_uris.get("assay", [])
    df["material_uri"] = df["material_idx"].apply(lambda i: mat_uris[i] if i < len(mat_uris) else "")
    df["assay_uri"] = df["assay_idx"].apply(lambda j: assay_uris[j] if j < len(assay_uris) else "")
    rdf_path = rdf_path or _find_rdf_in_mappings()
    uris_to_fetch = set(df["material_uri"].dropna().unique()) | set(df["assay_uri"].dropna().unique())
    uris_to_fetch.discard("")
    uri_to_label = _load_uri_to_label(rdf_path, list(uris_to_fetch)) if rdf_path else {}
    def _label(u, kind="Material"):
        if not u:
            return ""
        if u in uri_to_label and uri_to_label[u] != u.split("/")[-1].split("#")[-1]:
            return uri_to_label[u]
        frag = u.split("/")[-1].split("#")[-1] or u
        return f"{kind} {frag}" if frag.isdigit() or (frag and frag[0].isdigit()) else (uri_to_label.get(u, frag))
    df["material_label"] = df["material_uri"].map(lambda u: _label(u, "Material"))
    df["assay_label"] = df["assay_uri"].map(lambda u: _label(u, "Assay"))
    if source == "networkx":
        print("Note: Node URIs/labels from networkx_graph.pkl (index order should match if same graph as embeddings).")
    return df


def main():
    parser = argparse.ArgumentParser(description="Paper prediction report: high-confidence material-assay pairs and summary.")
    parser.add_argument("--csv", type=str, default=None, help="Use this existing CSV instead of loading torch (avoids segfault if torch.load crashes).")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="Max non-existing pairs to score (default 200000).")
    parser.add_argument("--threshold", type=float, default=0.98, help="Min probability to count (default 0.98; cos_sim >= 0.96).")
    parser.add_argument("--top-k", type=int, default=156, help="Max high-confidence pairs to output (default 156, paper number).")
    parser.add_argument("--no-diversify", action="store_true", help="Do not cap pairs per material; take top-k by score only (may repeat same material).")
    parser.add_argument("--rdf", type=str, default=None, help="RDF .ttl file for labels (default: mappings/NKB_RDF_V3.ttl or any .ttl in mappings/). Use --no-labels to skip.")
    parser.add_argument("--no-labels", action="store_true", help="Do not add URI/label columns (use indices only).")
    parser.add_argument("--nx-graph", type=str, default=None, help="Path to networkx_graph.pkl for node URIs if improved_hetero_data_node_uris.pkl is missing (default: repo root).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    threshold = args.threshold
    df = __import__("pandas").DataFrame()

    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"ERROR: --csv file not found: {args.csv}")
            print("Generate the CSV on a machine where PyTorch runs (e.g. Google Colab, cluster),")
            print("then run: python scripts/paper_prediction_report.py --csv <path-to-that-csv>")
            sys.exit(1)
        print(f"Using existing CSV: {args.csv}")
        df = __import__("pandas").read_csv(args.csv)
        if not df.empty and "rank" not in df.columns:
            df["rank"] = range(1, len(df) + 1)
        if not df.empty:
            if not args.no_labels:
                add_uris_and_labels(df, rdf_path=args.rdf if args.rdf else None, networkx_path=args.nx_graph)
                if "material_uri" not in df.columns:
                    print("Note: No node URIs found (tried improved_hetero_data_node_uris.pkl and networkx_graph.pkl). CSV has indices only.")
            out_csv = os.path.join(OUTPUT_DIR, "high_confidence_material_assay_pairs.csv")
            df.to_csv(out_csv, index=False)
            print(f"Wrote {out_csv} ({len(df)} rows)")
    else:
        df = run_material_assay_predictions(threshold=threshold, max_pairs_cap=args.max_pairs, batch_size=2000, top_k=args.top_k, diversify_across_materials=not args.no_diversify)
        out_csv = os.path.join(OUTPUT_DIR, "high_confidence_material_assay_pairs.csv")
        if not df.empty:
            if not args.no_labels:
                add_uris_and_labels(df, rdf_path=args.rdf if args.rdf else None, networkx_path=args.nx_graph)
                if "material_uri" not in df.columns:
                    print("Note: No node URIs found (tried improved_hetero_data_node_uris.pkl and networkx_graph.pkl). CSV has indices only.")
            df.to_csv(out_csv, index=False)
            print(f"Wrote {out_csv} ({len(df)} rows)")
        else:
            print("No high-confidence pairs found or data missing.")

    summary_path = os.path.join(OUTPUT_DIR, "prediction_section_summary.txt")
    n_pairs = len(df) if not df.empty else 0
    with open(summary_path, "w") as f:
        f.write("PREDICTION SECTION: REPRODUCIBILITY AND METHODS\n")
        f.write("=" * 60 + "\n\n")
        f.write("1) HIGH-CONFIDENCE MATERIAL–ASSAY PAIRS (NON-EXISTING RELATIONSHIPS)\n")
        f.write("   Count (score > {threshold}): {n_pairs} (target 156; capped at top-k, diversified across materials).\n")
        f.write("   Reproduce: python scripts/paper_prediction_report.py (uses embeddings only, no RDF).\n")
        f.write("   Output: high_confidence_material_assay_pairs.csv (material_idx, assay_idx, probability, similarity_score, rank, URIs, labels).\n\n")
        f.write("   What the score means: The 'probability' / 'similarity_score' column is embedding cosine similarity\n")
        f.write("   rescaled to [0,1]. It is NOT a calibrated link probability. Many pairs can have the same or very\n")
        f.write("   similar scores when their embeddings point in similar directions; that indicates similarity in\n")
        f.write("   embedding space, not necessarily equal likelihood of a true link. For calibrated probabilities,\n")
        f.write("   use the full R-GCN link predictor (with relation embeddings).\n\n")
        f.write("   'NOT PRESENT IN CURRENT LITERATURE': defined as not an edge in the NKB graph.\n\n")
        f.write("2) MATERIAL TYPES BY AGENCY: python scripts/agency_material_coverage_rdf.py (RDF, no Neo4j)\n")
        f.write("   Alternative (Neo4j): python scripts/agency_material_coverage.py\n\n")
        f.write("3) CLUSTERING / BIOLOGICAL MECHANISMS: see docs/PREDICTION_SECTION_METHODS.md.\n\n")
        f.write("=" * 60 + "\n")
        f.write("ITEMS FROM THE EMAIL TO ADDRESS/FIX (for Holly / co-authors)\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. REPORT NON-EXISTING RELATIONSHIPS / UNEXPLORED MATERIAL–ASSAY COMBINATIONS\n")
        f.write("   Retrieve: Run this script; output is high_confidence_material_assay_pairs.csv.\n")
        f.write("   Status: Addressed by this script and the CSV.\n\n")
        f.write("2. \"156 HIGH-CONFIDENCE PAIRS NOT PRESENT IN CURRENT LITERATURE\" — HOW DETERMINED?\n")
        f.write("   Definition: \"Not present in current literature\" = not an edge in the NKB knowledge graph.\n")
        f.write("   NKB aggregates published nanomaterial/toxicology studies; absence of a material–assay edge in NKB\n")
        f.write("   means that combination has not been reported in the curated literature. Add a Methods/Results sentence\n")
        f.write("   stating this (see docs/PREDICTION_SECTION_METHODS.md for suggested wording).\n\n")
        f.write("3. \"43 MATERIAL TYPES STUDIED BY ONE AGENCY BUT NOT OTHERS\" — REPORT IT\n")
        f.write("   Action: Run python scripts/agency_material_coverage_rdf.py (RDF, no Neo4j) or\n")
        f.write("   scripts/agency_material_coverage.py (Neo4j). Output: agency_material_coverage_report.txt.\n")
        f.write("   Report the number in the paper; if 43 came from a different definition, adjust the script and document.\n\n")
        f.write("4. \"CLUSTERING PATTERNS THAT SUGGEST SHARED BIOLOGICAL MECHANISMS\" — WHICH MECHANISMS?\n")
        f.write("   Clarification: The code does NOT infer named biological mechanisms. Clustering is by entity type and\n")
        f.write("   proximity in embedding space. Replace the phrase with one of the suggested wordings in\n")
        f.write("   docs/PREDICTION_SECTION_METHODS.md (Options A/B/C) so readers know it is interpretation, not model output.\n\n")
        f.write("CHECKLIST: Run paper_prediction_report.py; run agency_material_coverage_rdf.py (or agency_material_coverage.py); add Methods sentence for\n")
        f.write("\"not in literature\"; replace \"biological mechanisms\" with suggested wording. See docs/PREDICTION_SECTION_METHODS.md.\n")
    print(f"Wrote {summary_path}")
    print("\n--- Items from the email to address/fix (full text in prediction_section_summary.txt) ---")
    print("1. Report non-existing relationships: use high_confidence_material_assay_pairs.csv from this run.")
    print("2. \"Not in literature\": define as not an edge in NKB graph; add Methods sentence (see PREDICTION_SECTION_METHODS.md).")
    print("3. \"43 material types by agency\": run scripts/agency_material_coverage_rdf.py (RDF) or agency_material_coverage.py (Neo4j); report number.")
    print("4. \"Biological mechanisms\": use suggested wording in docs/PREDICTION_SECTION_METHODS.md (clustering is by embedding space, not named mechanisms).")
    print("---")


if __name__ == "__main__":
    main()
