#!/usr/bin/env python3
"""
Build a gold benchmark for GraphRAG evaluation with DETERMINISTIC ground truth.

Ground-truth answers are computed by direct Cypher queries against the loaded
Neo4j graph -- i.e. the "conventional database querying" the RAG is benchmarked
against. Each item records the canonical query, the human-facing gold answer,
and the gold entity set used for retrieval precision/recall.

Questions are generated from values that actually exist in the data (materials,
countries, assay endpoints) so every question is answerable.

Run (base env has neo4j; torch not needed):
    /Users/pranavsingh/miniforge3/bin/python scripts/build_graphrag_gold.py

Output: benchmarks/graphrag_gold.jsonl
"""

import argparse
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO_ROOT, "benchmarks", "graphrag_gold.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "ontosearcher"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from neo4j import GraphDatabase

    d = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    def run(q, **p):
        with d.session() as s:
            return s.run(q, **p).data()

    items = []
    _id = 0

    def add(**kw):
        nonlocal _id
        kw["id"] = f"q{_id:03d}"
        _id += 1
        items.append(kw)

    # ---- pick real materials (case-insensitive substrings that exist) ----
    materials = ["Silver", "Titanium", "Carbon", "Zinc", "Gold", "Copper", "Silica"]
    for mat in materials:
        rows = run(
            "MATCH (p) WHERE 'CPSC' IN p.graphs AND p.NPO_1808 IS NOT NULL "
            "AND toLower(p.NPO_1808) CONTAINS toLower($m) "
            "RETURN p.uri AS uri, p.NPO_1808 AS nano, p.C43530 AS mfr, p.C93401 AS ptype",
            m=mat,
        )
        if len(rows) < 3:
            continue
        uris = sorted({r["uri"] for r in rows})
        mfrs = sorted({r["mfr"] for r in rows if r.get("mfr")})
        # A) count
        add(category="count_material",
            question=f"How many CPSC consumer products contain {mat.lower()} nanomaterial?",
            eval_type="numeric",
            gold_answer=len(uris),
            gold_entities=uris,
            entity_field="product_uris",
            cypher=f"MATCH (p) WHERE 'CPSC' IN p.graphs AND toLower(p.NPO_1808) CONTAINS toLower('{mat}') RETURN count(p)")
        # B) manufacturers list
        if len(mfrs) >= 2:
            add(category="list_manufacturers",
                question=f"Which manufacturers make CPSC products containing {mat.lower()}?",
                eval_type="set",
                gold_answer=mfrs,
                gold_entities=mfrs,
                entity_field="manufacturers",
                cypher=f"MATCH (p) WHERE 'CPSC' IN p.graphs AND toLower(p.NPO_1808) CONTAINS toLower('{mat}') RETURN DISTINCT p.C43530")

    # ---- most common specified nanomaterial ----
    top = run(
        "MATCH (p) WHERE 'CPSC' IN p.graphs AND p.NPO_1808 IS NOT NULL "
        "AND NOT toLower(p.NPO_1808) IN ['unknown','unspecified nanomaterials','not specified','none','n/a'] "
        "WITH p.NPO_1808 AS m, count(*) AS c RETURN m, c ORDER BY c DESC LIMIT 1")
    if top:
        add(category="top_nanomaterial",
            question="What is the most common specified nanomaterial among CPSC consumer products (excluding 'Unknown' and 'Unspecified')?",
            eval_type="contains_text",
            gold_answer=top[0]["m"],
            gold_entities=[top[0]["m"]],
            entity_field="nanomaterials",
            cypher="... ORDER BY count DESC LIMIT 1 (excluding Unknown/Unspecified)")

    # ---- existence by country + material ----
    countries = run("MATCH (p) WHERE 'CPSC' IN p.graphs AND p.C25464 IS NOT NULL "
                    "WITH p.C25464 AS c, count(*) AS n RETURN c ORDER BY n DESC LIMIT 4")
    for c in [r["c"] for r in countries]:
        rows = run("MATCH (p) WHERE 'CPSC' IN p.graphs AND p.C25464 = $c "
                   "AND toLower(p.NPO_1808) CONTAINS 'silver' RETURN p.uri AS uri", c=c)
        add(category="exists_country_material",
            question=f"Are there any CPSC products from {c} that contain silver?",
            eval_type="boolean",
            gold_answer=bool(rows),
            gold_entities=sorted({r["uri"] for r in rows}),
            entity_field="product_uris",
            cypher=f"MATCH (p) WHERE 'CPSC' IN p.graphs AND p.C25464='{c}' AND toLower(p.NPO_1808) CONTAINS 'silver' RETURN count(p)>0")

    # ---- assay endpoints ----
    endpoints = ["LDH", "Cytokine", "viability", "genotox", "apoptosis", "ROS", "inflammation"]
    for ep in endpoints:
        rows = run(
            "MATCH (a:Assay) WHERE 'NIOSH' IN a.graphs AND "
            "((a.label IS NOT NULL AND toLower(a.label) CONTAINS toLower($e)) OR "
            " (a.description IS NOT NULL AND toLower(a.description) CONTAINS toLower($e))) "
            "RETURN a.uri AS uri, a.label AS name", e=ep)
        if len(rows) < 1:
            continue
        uris = sorted({r["uri"] for r in rows})
        names = sorted({r["name"] for r in rows if r.get("name")})
        add(category="count_assay",
            question=f"How many NIOSH toxicology assays are related to {ep}?",
            eval_type="numeric",
            gold_answer=len(uris),
            gold_entities=uris,
            entity_field="assay_uris",
            cypher=f"MATCH (a:Assay) WHERE 'NIOSH' IN a.graphs AND (toLower(a.label) CONTAINS toLower('{ep}') OR toLower(a.description) CONTAINS toLower('{ep}')) RETURN count(a)")
        add(category="list_assays",
            question=f"Which toxicology assay endpoints in the NIOSH data relate to {ep}?",
            eval_type="set",
            gold_answer=names,
            gold_entities=names,
            entity_field="assay_names",
            cypher=f"... RETURN DISTINCT a.label for endpoint '{ep}'")

    d.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(items)} gold items to {args.out}")
    from collections import Counter
    print("By category:", dict(Counter(it["category"] for it in items)))


if __name__ == "__main__":
    main()
