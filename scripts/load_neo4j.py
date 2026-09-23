#!/usr/bin/env python3
"""
Load the CPSC / NIOSH / NKB RDF (Turtle) files into Neo4j using the exact
property-graph schema the GraphRAG queries in ``src/rag`` expect.

This reproduces (cleanly) the transform used in the original
``notebooks/experiments/llmexperiment.ipynb`` ComplexRDFLoader:

* One node per RDF subject URI, base label ``Resource`` + ``uri`` property.
* ``graphs`` list property records the source(s) ("CPSC", "NIOSH", "NKB").
* Node labels derived from ``rdf:type``:
    - ``obo:OBI_0000070``  -> ``Assay``      (NIOSH assay nodes)
    - ``npo:NPO_199``      -> ``Product``     (CPSC products)
    - the type's local name is also added as a label.
* Literal object of each predicate becomes a property named by the predicate's
  local name (e.g. ``npo:NPO_1808`` -> ``NPO_1808``, ``rdfs:label`` -> ``label``,
  ``dcterms:description`` -> ``description``). Multiple literals for the same
  predicate are joined with "; ".
* Object URIs are stored as relationships (local-name relation type) so the
  graph is navigable, but the RAG queries only rely on the literal properties.

The RAG Cypher then works unchanged, e.g. CPSC:
    MATCH (product) WHERE 'CPSC' IN product.graphs AND product.NPO_1808 CONTAINS $t
and NIOSH:
    MATCH (assay) WHERE 'NIOSH' IN assay.graphs AND 'Assay' IN labels(assay) ...

Run (use an env with neo4j + rdflib; base env works, torch is not needed):
    /Users/pranavsingh/miniforge3/bin/python scripts/load_neo4j.py \
        --sources cpsc niosh

Add ``nkb`` to also load the large NKB graph (slow; ~1.37M triples).
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAPPINGS = os.path.join(REPO_ROOT, "mappings")

SOURCE_FILES = {
    "cpsc": os.path.join(MAPPINGS, "cpsc_database.ttl"),
    "niosh": os.path.join(MAPPINGS, "niosh_rdf_tutV2c.ttl"),
    "nkb": os.path.join(MAPPINGS, "NKB_RDF_V3.ttl"),
}

# rdf:type local-name -> extra Neo4j label
TYPE_LABEL_MAP = {
    "OBI_0000070": "Assay",
    "NPO_199": "Product",
}


def clean_name(uri: str) -> str:
    """Local name after '#' or '/', sanitized for Neo4j identifiers."""
    last = uri.split("#")[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", last)
    if name and name[0].isdigit():
        name = "p_" + name
    return name or "value"


def load_source(driver, source: str, path: str, batch_size: int = 1000):
    import rdflib
    from rdflib import BNode, Literal, RDF, URIRef

    print(f"\n=== Loading {source.upper()} from {path} ===")
    if not os.path.isfile(path):
        print(f"  ERROR: file not found: {path}")
        return 0
    g = rdflib.Graph()
    t0 = time.time()
    g.parse(path, format="turtle")
    print(f"  parsed {len(g):,} triples in {time.time()-t0:.1f}s")

    # Aggregate per-subject: labels, literal properties, object-URI relations.
    props = defaultdict(dict)          # uri -> {prop: value}
    multi = defaultdict(lambda: defaultdict(list))  # uri -> prop -> [values]
    labels = defaultdict(set)          # uri -> {labels}
    rels = []                          # (from_uri, rel_type, to_uri)

    for s, p, o in g:
        if not isinstance(s, URIRef):
            continue  # skip blank-node subjects (not queried by the RAG)
        su = str(s)
        labels[su].add("Resource")
        if p == RDF.type and isinstance(o, URIRef):
            ln = clean_name(str(o))
            if ln in TYPE_LABEL_MAP:
                labels[su].add(TYPE_LABEL_MAP[ln])
            labels[su].add(ln)
            continue
        pn = clean_name(str(p))
        if isinstance(o, Literal):
            multi[su][pn].append(str(o))
        elif isinstance(o, URIRef):
            rels.append((su, pn, str(o)))
        # blank-node objects are ignored (their descriptions are not queried)

    for su, pdict in multi.items():
        for pn, vals in pdict.items():
            props[su][pn] = "; ".join(dict.fromkeys(vals)) if len(vals) > 1 else vals[0]

    all_uris = set(labels)
    print(f"  subjects: {len(all_uris):,} | literal-prop nodes: {len(props):,} | relations: {len(rels):,}")

    # ---- write nodes in batches ----
    node_rows = []
    for su in all_uris:
        lset = labels[su] - {"Resource"}
        node_rows.append({
            "uri": su,
            "graph": source.upper(),
            "labels": sorted(lset),
            "props": props.get(su, {}),
        })

    cypher_nodes = """
    UNWIND $rows AS row
    MERGE (n:Resource {uri: row.uri})
    SET n += row.props
    SET n.graphs = CASE
        WHEN n.graphs IS NULL THEN [row.graph]
        WHEN NOT row.graph IN n.graphs THEN n.graphs + row.graph
        ELSE n.graphs END
    WITH n, row
    CALL apoc.create.addLabels(n, row.labels) YIELD node
    RETURN count(node)
    """
    # apoc may not be installed; fall back to per-label SET via dynamic labels
    cypher_nodes_noapoc = """
    UNWIND $rows AS row
    MERGE (n:Resource {uri: row.uri})
    SET n += row.props
    SET n.graphs = CASE
        WHEN n.graphs IS NULL THEN [row.graph]
        WHEN NOT row.graph IN n.graphs THEN n.graphs + row.graph
        ELSE n.graphs END
    """

    use_apoc = _has_apoc(driver)
    t0 = time.time()
    with driver.session() as sess:
        sess.run("CREATE CONSTRAINT resource_uri IF NOT EXISTS FOR (n:Resource) REQUIRE n.uri IS UNIQUE")
        for i in range(0, len(node_rows), batch_size):
            batch = node_rows[i:i + batch_size]
            if use_apoc:
                sess.run(cypher_nodes, rows=batch)
            else:
                sess.run(cypher_nodes_noapoc, rows=batch)
        if not use_apoc:
            _apply_labels_noapoc(sess, node_rows, batch_size)
    print(f"  wrote {len(node_rows):,} nodes in {time.time()-t0:.1f}s (apoc={use_apoc})")

    # ---- write relationships in batches (generic RELATED edge with rtype prop) ----
    cypher_rels = """
    UNWIND $rows AS row
    MATCH (a:Resource {uri: row.from})
    MERGE (b:Resource {uri: row.to})
    ON CREATE SET b.graphs = [row.graph]
    MERGE (a)-[r:RELATED {rtype: row.rtype}]->(b)
    """
    t0 = time.time()
    with driver.session() as sess:
        rel_rows = [{"from": f, "rtype": rt, "to": t, "graph": source.upper()} for f, rt, t in rels]
        for i in range(0, len(rel_rows), batch_size):
            sess.run(cypher_rels, rows=rel_rows[i:i + batch_size])
    print(f"  wrote {len(rels):,} relationships in {time.time()-t0:.1f}s")
    return len(node_rows)


def _has_apoc(driver) -> bool:
    try:
        with driver.session() as sess:
            sess.run("RETURN apoc.version()").single()
        return True
    except Exception:
        return False


def _apply_labels_noapoc(sess, node_rows, batch_size):
    """Group by label set and SET labels via a generated Cypher statement."""
    by_labels = defaultdict(list)
    for row in node_rows:
        if row["labels"]:
            by_labels[tuple(row["labels"])].append(row["uri"])
    for lset, uris in by_labels.items():
        label_str = ":".join(f"`{l}`" for l in lset)
        q = f"UNWIND $uris AS u MATCH (n:Resource {{uri: u}}) SET n:{label_str}"
        for i in range(0, len(uris), batch_size):
            sess.run(q, uris=uris[i:i + batch_size])


def summarize(driver):
    print("\n=== Loaded graph summary ===")
    with driver.session() as sess:
        total = sess.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        print(f"  total nodes: {total:,}")
        for src in ("CPSC", "NIOSH", "NKB"):
            c = sess.run("MATCH (n) WHERE $s IN n.graphs RETURN count(n) AS c", s=src).single()["c"]
            if c:
                print(f"  {src}: {c:,} nodes")
        prod = sess.run("MATCH (n) WHERE 'CPSC' IN n.graphs AND n.NPO_1808 IS NOT NULL RETURN count(n) AS c").single()["c"]
        assays = sess.run("MATCH (n:Assay) WHERE 'NIOSH' IN n.graphs RETURN count(n) AS c").single()["c"]
        print(f"  CPSC products with NPO_1808: {prod:,}")
        print(f"  NIOSH Assay nodes: {assays:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "ontosearcher"))
    ap.add_argument("--sources", nargs="+", default=["cpsc", "niosh"], choices=list(SOURCE_FILES))
    ap.add_argument("--wipe", action="store_true", help="Delete all nodes before loading.")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {args.uri}")

    if args.wipe:
        print("Wiping existing data...")
        with driver.session() as sess:
            sess.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")

    for src in args.sources:
        load_source(driver, src, SOURCE_FILES[src], batch_size=args.batch_size)

    summarize(driver)
    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
