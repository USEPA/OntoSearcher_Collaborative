#!/usr/bin/env python3
"""
Report material types (e.g. nanomaterial names) that appear in one federal dataset
(CPSC, NIOSH, or NKB) but not in the others. Used for the paper claim:
"43 material types studied by one agency but not others."

Requires Neo4j running with the combined nanotoxicology graph and node property
that records source (e.g. product.graphs / assay.graphs containing 'CPSC', 'NIOSH', 'NKB').

Run from repo root:
  python scripts/agency_material_coverage.py

For RDF-only (no Neo4j), use: python scripts/agency_material_coverage_rdf.py

Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD if not using defaults.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Install neo4j: pip install neo4j")
        return 1

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "ontosearcher")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Material-type property: CPSC uses NPO_1808 (nanomaterial); NKB/NIOSH may use label/description.
    # Adjust property names to match your Neo4j schema.
    queries = {
        "CPSC_materials": """
        MATCH (n)
        WHERE 'CPSC' IN n.graphs AND n.NPO_1808 IS NOT NULL
        RETURN DISTINCT n.NPO_1808 AS material
        """,
        "NIOSH_materials": """
        MATCH (n)
        WHERE 'NIOSH' IN n.graphs AND (n.label IS NOT NULL OR n.description IS NOT NULL)
        RETURN DISTINCT coalesce(n.label, n.description) AS material
        LIMIT 5000
        """,
        "NKB_materials": """
        MATCH (n)
        WHERE n.graphs IS NOT NULL AND (n.label IS NOT NULL OR n.NPO_1808 IS NOT NULL)
        AND NOT 'CPSC' IN n.graphs AND NOT 'NIOSH' IN n.graphs
        RETURN DISTINCT coalesce(n.NPO_1808, n.label) AS material
        LIMIT 5000
        """
    }

    with driver.session() as session:
        sets = {}
        for name, q in queries.items():
            try:
                result = session.run(q)
                sets[name] = set(r["material"] for r in result if r["material"])
            except Exception as e:
                print(f"Query {name} failed: {e}")
                sets[name] = set()

    driver.close()

    # Normalize for comparison (e.g. strip, lower)
    def norm(s):
        return (s or "").strip().lower()

    cpsc = {norm(m) for m in sets.get("CPSC_materials", [])}
    niosh = {norm(m) for m in sets.get("NIOSH_materials", [])}
    nkb = {norm(m) for m in sets.get("NKB_materials", [])}

    only_cpsc = cpsc - niosh - nkb
    only_niosh = niosh - cpsc - nkb
    only_nkb = nkb - cpsc - niosh

    total_agency_specific = len(only_cpsc) + len(only_niosh) + len(only_nkb)

    out_path = os.path.join(REPO_ROOT, "agency_material_coverage_report.txt")
    with open(out_path, "w") as f:
        f.write("Material types appearing in one agency/source but not others\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Only in CPSC: {len(only_cpsc)}\n")
        f.write(f"Only in NIOSH: {len(only_niosh)}\n")
        f.write(f"Only in NKB (non-CPSC/NIOSH): {len(only_nkb)}\n")
        f.write(f"Total distinct material types in exactly one source: {total_agency_specific}\n\n")
        f.write("(Adjust queries in this script if your Neo4j schema uses different property names or graph labels.)\n")
        if only_cpsc:
            f.write("\nSample (CPSC-only): " + ", ".join(list(only_cpsc)[:20]) + "\n")
        if only_niosh:
            f.write("Sample (NIOSH-only): " + ", ".join(list(only_niosh)[:20]) + "\n")
        if only_nkb:
            f.write("Sample (NKB-only): " + ", ".join(list(only_nkb)[:20]) + "\n")

    print(f"Wrote {out_path}")
    print(f"Total material types in exactly one agency: {total_agency_specific}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
