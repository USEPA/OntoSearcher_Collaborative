#!/usr/bin/env python3
"""
Report material types that appear in one federal dataset (CPSC, NIOSH, or NKB)
but not in the others. Uses RDF files only (no Neo4j). For the paper claim:
"Material types studied by one agency but not others."

Run from repo root:
  python scripts/agency_material_coverage_rdf.py

Optional:
  --mappings DIR   Directory containing NKB_RDF_V3.ttl, cpsc_database.ttl, niosh_rdf_tutV2c.ttl (default: mappings/)
  --out FILE       Output report path (default: agency_material_coverage_report.txt)
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RDF predicate URIs
NPO_1808 = "http://purl.bioontology.org/ontology/npo#NPO_1808"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
NCIT_C25365 = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C25365"
NCIT_C90353 = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C90353"
NCIT_C93410 = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C93410"
DCTERMS_TYPE = "http://purl.org/dc/terms/type"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _get_fragment(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.split("/")[-1]


def _collect_cpsc_materials(graph) -> set:
    """CPSC: material type = value of npo:NPO_1808 (literal)."""
    from rdflib import Literal
    out = set()
    for _s, _p, _o in graph.triples((None, None, None)):
        if str(_p) == NPO_1808 and isinstance(_o, Literal):
            out.add(_norm(str(_o)))
    return out


def _collect_niosh_materials(graph) -> set:
    """NIOSH: material type = ncit:C90353 (preferred) or ncit:C93410 or dcterms:type."""
    from rdflib import Literal
    out = set()
    mat_prefix = "http://example.org/materials/row/"
    for s, p, o in graph.triples((None, None, None)):
        subj = str(s)
        if not subj.startswith(mat_prefix):
            continue
        if isinstance(o, Literal):
            if str(p) == NCIT_C90353:
                out.add(_norm(str(o)))
            elif str(p) == NCIT_C93410:
                out.add(_norm(str(o)))
            elif str(p) == DCTERMS_TYPE:
                out.add(_norm(str(o)))
    return out


def _collect_nkb_materials(graph) -> set:
    """NKB: material type = object of npo:NPO_1808 (URI → fragment or label), or ncit:C25365 snippet, or material URI id."""
    from rdflib import Literal, URIRef
    out = set()
    mat_prefix = "http://example.org/material/"
    # 1) npo:NPO_1808: object can be URI (ncit:C61975) or literal
    for s, p, o in graph.triples((None, None, None)):
        if str(p) != NPO_1808:
            continue
        subj = str(s)
        if not subj.startswith(mat_prefix):
            continue
        if isinstance(o, Literal):
            out.add(_norm(str(o)))
        elif isinstance(o, URIRef):
            label = None
            for _o in graph.objects(o, None):
                if isinstance(_o, Literal) and str(_o).strip():
                    label = _norm(str(_o))
                    break
            if label:
                out.add(label)
            else:
                out.add(_norm(_get_fragment(str(o))))
    # 2) Materials with no NPO_1808: use ncit:C25365 first phrase or "material {id}"
    for s in graph.subjects(None, None):
        subj = str(s)
        if not subj.startswith(mat_prefix):
            continue
        has_npo = any(1 for _ in graph.objects(s, URIRef(NPO_1808)))
        if has_npo:
            continue
        label = None
        for o in graph.objects(s, URIRef(NCIT_C25365)):
            if isinstance(o, Literal):
                raw = str(o).strip()
                if raw:
                    m = re.match(r"^(?:this nano scaled material is composed of (?:a |an )?([^,.]+)", raw, re.I)
                    label = _norm(m.group(1)) if m else _norm(raw[:60])
                break
        if label:
            out.add(label)
        else:
            frag = subj.split("/")[-1].split("#")[-1]
            if frag.isdigit():
                out.add("material " + frag)
    return out


def main():
    parser = argparse.ArgumentParser(description="Agency material coverage from RDF (no Neo4j).")
    parser.add_argument("--mappings", type=str, default=os.path.join(REPO_ROOT, "mappings"),
                        help="Directory with NKB_RDF_V3.ttl, cpsc_database.ttl, niosh_rdf_tutV2c.ttl")
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "agency_material_coverage_report.txt"),
                        help="Output report path")
    args = parser.parse_args()

    try:
        from rdflib import Graph, Literal, URIRef
    except ImportError:
        print("Install rdflib: pip install rdflib")
        return 1

    mappings_dir = args.mappings
    paths = {
        "NKB": os.path.join(mappings_dir, "NKB_RDF_V3.ttl"),
        "CPSC": os.path.join(mappings_dir, "cpsc_database.ttl"),
        "NIOSH": os.path.join(mappings_dir, "niosh_rdf_tutV2c.ttl"),
    }
    sets = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            print(f"Skip {name}: file not found {path}")
            sets[name] = set()
            continue
        g = Graph()
        try:
            g.parse(path, format="turtle")
        except Exception as e:
            print(f"Parse error {name} ({path}): {e}")
            sets[name] = set()
            continue
        if name == "CPSC":
            sets[name] = _collect_cpsc_materials(g)
        elif name == "NIOSH":
            sets[name] = _collect_niosh_materials(g)
        else:
            sets[name] = _collect_nkb_materials(g)
        print(f"  {name}: {len(sets[name])} distinct material types")

    cpsc = sets.get("CPSC", set())
    niosh = sets.get("NIOSH", set())
    nkb = sets.get("NKB", set())

    only_cpsc = cpsc - niosh - nkb
    only_niosh = niosh - cpsc - nkb
    only_nkb = nkb - cpsc - niosh
    total_agency_specific = len(only_cpsc) + len(only_niosh) + len(only_nkb)

    out_path = args.out
    with open(out_path, "w") as f:
        f.write("Material types appearing in one agency/source but not others (RDF-based, no Neo4j)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Sources: NKB_RDF_V3.ttl, cpsc_database.ttl, niosh_rdf_tutV2c.ttl\n\n")
        f.write(f"Only in CPSC: {len(only_cpsc)}\n")
        f.write(f"Only in NIOSH: {len(only_niosh)}\n")
        f.write(f"Only in NKB: {len(only_nkb)}\n")
        f.write(f"Total distinct material types in exactly one source: {total_agency_specific}\n\n")
        if only_cpsc:
            f.write("Sample (CPSC-only): " + ", ".join(sorted(only_cpsc)[:25]) + "\n")
        if only_niosh:
            f.write("Sample (NIOSH-only): " + ", ".join(sorted(only_niosh)[:25]) + "\n")
        if only_nkb:
            f.write("Sample (NKB-only): " + ", ".join(sorted(only_nkb)[:25]) + "\n")

    print(f"Wrote {out_path}")
    print(f"Total material types in exactly one agency: {total_agency_specific}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
