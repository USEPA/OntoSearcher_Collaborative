"""
Nanotoxicology RAG: same retrieval logic as llmexperiment.ipynb, with a pluggable LLM backend.
Supports OpenAI or a local model (e.g. Llama 2 7B) via the transformers library.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional

from .llm_backends import LLMBackend

logger = logging.getLogger("nanotox_rag")


class NanotoxicologyRAG:
    """
    RAG system for nanotoxicology: Neo4j retrieval + LLM generation.
    LLM can be OpenAI or a local model (e.g. Llama 7B) via the passed backend.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_backend: LLMBackend,
        verbose: bool = False,
    ):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = llm_backend
        self.verbose = verbose
        # Structured query trace (list of dicts). Reset per answer_question_traced call.
        self.query_trace: List[Dict[str, Any]] = []

    def close(self) -> None:
        self.driver.close()

    def run_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        if params is None:
            params = {}
        if self.verbose:
            print("\nCypher Query:")
            print(query)
            if params:
                print("\nParameters:")
                print(json.dumps(params, indent=2))
        t0 = time.perf_counter()
        error = None
        results: List[Dict] = []
        try:
            with self.driver.session() as session:
                results = session.run(query, **params).data()
        except Exception as e:  # log and re-raise so callers keep their fallbacks
            error = str(e)
            logger.error("Cypher query failed: %s | params=%s", error, params)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.query_trace.append({
                "query": " ".join(query.split()),
                "params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                           for k, v in params.items()},
                "n_results": len(results),
                "elapsed_ms": round(elapsed_ms, 2),
                "error": error,
            })
            logger.info("cypher n_results=%d elapsed_ms=%.1f params=%s",
                        len(results), elapsed_ms, params)
        if self.verbose:
            print(f"\nQuery returned {len(results)} results")
        return results

    # ---------- Cross-graph query templates ----------
    # Every source graph is searched the same way: any node in that graph
    # (no entity-type filter) whose *text* properties contain the term.
    # Property lists are the string fields actually present on each graph;
    # numeric-only measurement fields (SIO_000300, value, years) are omitted
    # so a term cannot accidentally match a number.
    GRAPHS = ("CPSC", "NIOSH", "NKB")
    _TEXT_PROPS = {
        "CPSC": [
            "NPO_1808", "C93401", "C43530", "C93400", "C25464",
            "description", "CHMO_0000101", "label",
        ],
        "NIOSH": [
            "label", "description", "comment", "type", "C93410",
            "hasRelatedSynonym", "hasExactSynonym", "P90",
        ],
        "NKB": [
            "label", "NPO_1808", "C43530", "C42614", "C25365", "C25704",
            "C25480", "C60765", "C42774", "IAO_0000630", "P90",
            "hasExactSynonym", "C25372", "C68553",
        ],
    }

    @staticmethod
    def _prop_match(alias: str, props: List[str]) -> str:
        return " OR\n          ".join(
            f"({alias}.{p} IS NOT NULL AND toLower(toString({alias}.{p})) "
            f"CONTAINS toLower($search_term))"
            for p in props
        )

    def _graph_where(self, alias: str, graph: str) -> str:
        """WHERE fragment: any entity type in ``graph`` matching on text props."""
        match = self._prop_match(alias, self._TEXT_PROPS[graph])
        return f"""
        WHERE '{graph}' IN {alias}.graphs
        AND (
          $search_term IS NULL OR $search_term = '' OR
          {match}
        )
        """

    def query_graph(self, graph: str, search_term: Optional[str] = None,
                    limit: int = 200) -> List[Dict]:
        """Return matching nodes of *any* entity type in one source graph."""
        where = self._graph_where("n", graph)
        query = f"""
        MATCH (n)
        {where}
        RETURN
          n.uri AS uri,
          labels(n) AS node_labels,
          coalesce(n.label, n.NPO_1808, n.description, n.C93410, n.C42614, n.type) AS name,
          n.NPO_1808 AS nanomaterial,
          n.C43530 AS manufacturer,
          n.C93401 AS product_type,
          n.C25464 AS country,
          n.C93400 AS category,
          n.description AS description,
          n.comment AS comment,
          n.type AS type,
          n.C93410 AS material_id,
          n.C25372 AS assay_type,
          n.C42614 AS measurement,
          n.C25365 AS medium,
          n.SIO_000300 AS value,
          n.SIO_000221 AS unit
        LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def count_graph(self, graph: str, search_term: Optional[str] = None) -> int:
        """Exact match count for one graph (all entity types; not LIMIT-capped)."""
        where = self._graph_where("n", graph)
        rows = self.run_query(
            f"MATCH (n) {where} RETURN count(n) AS c",
            {"search_term": search_term},
        )
        return int(rows[0]["c"]) if rows else 0

    def count_graph_by_label(self, graph: str, search_term: Optional[str] = None,
                             limit: int = 15) -> List[Dict]:
        """How many matches per Neo4j label (entity type) in one graph."""
        where = self._graph_where("n", graph)
        query = f"""
        MATCH (n)
        {where}
        UNWIND labels(n) AS l
        WITH l WHERE l <> 'Resource'
        RETURN l AS entity_type, count(*) AS count
        ORDER BY count DESC LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def distinct_graph_field(self, graph: str, search_term: str, prop: str) -> List[str]:
        """Distinct values of ``prop`` among matches in one graph."""
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", prop):
            return []
        where = self._graph_where("n", graph)
        query = f"""
        MATCH (n)
        {where}
        AND n.{prop} IS NOT NULL
        RETURN DISTINCT n.{prop} AS value
        """
        return [r["value"] for r in self.run_query(query, {"search_term": search_term})]

    # Back-compat wrappers used by the eval harness / older callers.
    def query_cpsc_products(self, search_term: Optional[str] = None, limit: int = 200) -> List[Dict]:
        return self.query_graph("CPSC", search_term, limit)

    def count_cpsc_products(self, search_term: Optional[str] = None) -> int:
        return self.count_graph("CPSC", search_term)

    def query_niosh_assays(self, search_term: Optional[str] = None, limit: int = 200) -> List[Dict]:
        rows = self.query_graph("NIOSH", search_term, limit)
        for r in rows:
            r.setdefault("name", r.get("name") or r.get("type") or r.get("material_id"))
        return rows

    def count_niosh_assays(self, search_term: Optional[str] = None) -> int:
        return self.count_graph("NIOSH", search_term)

    def query_nkb_materials(self, search_term: Optional[str] = None, limit: int = 200) -> List[Dict]:
        return self.query_graph("NKB", search_term, limit)

    def count_nkb_matches(self, search_term: Optional[str] = None) -> int:
        return self.count_graph("NKB", search_term)

    def distinct_cpsc_field(self, search_term: str, field: str) -> List[str]:
        """Distinct values of a CPSC property (e.g. C43530 manufacturers) for a term."""
        allowed = {"manufacturer": "C43530", "product_type": "C93401",
                   "category": "C93400", "nanomaterial": "NPO_1808",
                   "country": "C25464", "uri": "uri"}
        return self.distinct_graph_field("CPSC", search_term, allowed.get(field, field))

    def cpsc_country_breakdown(self, search_term: str, limit: int = 25) -> List[Dict]:
        """Country distribution (C25464) among CPSC records matching a term."""
        where = self._graph_where("product", "CPSC")
        query = f"""
        MATCH (product)
        {where}
        AND product.C25464 IS NOT NULL
        WITH product.C25464 AS country, count(*) AS count
        RETURN country, count ORDER BY count DESC LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def distinct_niosh_labels(self, search_term: str) -> List[str]:
        return self.distinct_graph_field("NIOSH", search_term, "label")

    def distinct_nkb_materials(self, search_term: str) -> List[str]:
        mats = self.distinct_graph_field("NKB", search_term, "NPO_1808")
        labs = self.distinct_graph_field("NKB", search_term, "label")
        seen, out = set(), []
        for v in mats + labs:
            key = str(v).lower()
            if v and key not in seen:
                seen.add(key)
                out.append(v)
        return out

    def _deduplicate_by_uri(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for item in items:
            uri = item.get("uri")
            if uri and uri not in seen:
                seen.add(uri)
                out.append(item)
        return out

    # ---------- RAG: question analysis (uses LLM; fallback if JSON fails) ----------

    def analyze_question(self, question: str) -> Dict[str, List[str]]:
        """Determine search criteria from the question using the LLM; fallback to simple keyword extraction."""
        prompt = f"""
Analyze this question about nanotoxicology and determine the key search criteria.

Question: {question}

Identify:
1. Mentioned nanomaterials (e.g., silver, titanium dioxide, carbon nanotubes)
2. Product types or categories (e.g., baby products, kitchenware, air filters)
3. Toxicology assays or endpoints (e.g., cytotoxicity, genotoxicity, LDH assay)
4. Exposure routes (e.g., inhalation, dermal)
5. Target organs or organisms (e.g., lung, skin, mice)

Return ONLY valid JSON with these keys:
{{"nanomaterials": [...], "products": [...], "assays": [...], "exposure_routes": [...], "targets": [...]}}
"""
        messages = [
            {"role": "system", "content": "You are a nanotoxicology expert assistant. Reply only with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.llm.generate(messages, max_new_tokens=512, temperature=0.3)
            text = text.strip()
            # Extract JSON (model may wrap in markdown or add text)
            start = text.find("{")
            if start >= 0:
                depth = 0
                for i, c in enumerate(text[start:], start=start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            text = text[start : i + 1]
                            break
            data = json.loads(text)
            return {
                "nanomaterials": _ensure_list(data.get("nanomaterials")),
                "products": _ensure_list(data.get("products")),
                "assays": _ensure_list(data.get("assays")),
                "exposure_routes": _ensure_list(data.get("exposure_routes")),
                "targets": _ensure_list(data.get("targets")),
            }
        except Exception as e:
            if self.verbose:
                print(f"LLM question analysis failed ({e}), using keyword fallback.")
            return _keyword_fallback_analysis(question)

    # Generic tokens that are not useful search terms on their own (they either
    # match everything or nothing). Terms made up only of these are dropped.
    _GENERIC_TOKENS = {
        "product", "products", "cpsc", "niosh", "nkb", "consumer", "consumers",
        "nanomaterial", "nanomaterials", "material", "materials", "nanoparticle",
        "nanoparticles", "assay", "assays", "data", "information", "database",
        "the", "a", "an", "of", "from", "in", "with", "that", "contain",
        "contains", "containing", "any", "all", "and", "or", "for",
    }

    @classmethod
    def _clean_terms(cls, *groups) -> List[str]:
        """Flatten, strip, dedupe (case-insensitive), and drop noisy terms.

        Filters out empty/very short terms, purely generic terms (e.g. 'products',
        'nanomaterial'), and long free-text phrases (>3 words such as
        'CPSC products from USA') that the LLM sometimes emits — these pollute the
        search with 0-match lookups and confuse downstream answering.
        """
        seen, out = set(), []

        def _add(term: str):
            key = term.lower()
            if len(term) >= 2 and key not in seen:
                seen.add(key)
                out.append(term)

        for grp in groups:
            for t in (grp or []):
                t = str(t).strip().strip(".,;:\"'")
                if len(t) < 2:
                    continue
                words = t.split()
                if len(words) > 3:
                    continue  # free-text phrase, not an entity keyword
                non_generic = [w for w in words if w.lower() not in cls._GENERIC_TOKENS]
                if not non_generic:
                    continue  # nothing but generic filler
                _add(t)
                # Also search the meaningful remainder so an over-specific phrase
                # like "silver nanoparticles" still matches records stored as
                # "Silver" (addresses the keyword-granularity failure mode).
                if len(non_generic) < len(words):
                    _add(" ".join(non_generic))
        return out

    def retrieve_relevant_data(self, analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """Retrieve Neo4j data based on analysis.

        Robust search: case-insensitive matching, exact match COUNTS (for
        'how many' questions), full DISTINCT value lists (for list questions,
        not capped by LIMIT), and failsafes when the LLM analysis is empty or a
        term returns nothing.
        """
        results: Dict[str, Any] = {
            "cpsc_products": [], "niosh_assays": [], "nkb_materials": [], "combined_data": [],
            "nanomaterial_stats": [], "match_counts": {}, "entity_type_counts": {},
            "distinct_values": {"manufacturers": [], "product_types": [], "nanomaterials": [],
                                "product_uris": [], "assay_names": [], "assay_uris": [],
                                "nkb_materials": [], "niosh_materials": [], "nkb_measurements": []},
            "country_breakdown": {},
            "search_terms": {"cpsc": [], "niosh": [], "nkb": [], "all": []},
        }

        # Same cleaned terms are sent to every graph so a material question
        # also hits NIOSH/NKB and an assay question also hits CPSC/NKB.
        # Exposure/target words (skin, lung, dermal) are too generic and
        # flood unrelated products, so they are not used as search terms.
        all_terms = self._clean_terms(
            analysis.get("nanomaterials"), analysis.get("products"),
            analysis.get("assays"),
        )
        results["search_terms"] = {"cpsc": all_terms, "niosh": all_terms,
                                   "nkb": all_terms, "all": all_terms}

        def _safe(fn, *a, default=None):
            try:
                return fn(*a)
            except Exception as e:
                logger.warning("retrieval sub-query failed: %s", e)
                return default if default is not None else []

        cpsc_uris, mfrs, ptypes, nanos = set(), set(), set(), set()
        assay_uris, assay_names, niosh_mats = set(), set(), set()
        nkb_mats, nkb_meas = set(), set()

        for term in all_terms:
            counts: Dict[str, int] = {}
            type_counts: Dict[str, List[Dict]] = {}

            # CPSC — all entity types (mostly Product)
            rows = _safe(self.query_graph, "CPSC", term)
            results["cpsc_products"].extend(rows)
            counts["cpsc"] = _safe(self.count_graph, "CPSC", term, default=0)
            if counts["cpsc"]:
                type_counts["CPSC"] = _safe(self.count_graph_by_label, "CPSC", term, default=[])
                mfrs.update(_safe(self.distinct_cpsc_field, term, "manufacturer"))
                ptypes.update(_safe(self.distinct_cpsc_field, term, "product_type"))
                nanos.update(_safe(self.distinct_cpsc_field, term, "nanomaterial"))
                cpsc_uris.update(_safe(self.distinct_cpsc_field, term, "uri"))
                cb = _safe(self.cpsc_country_breakdown, term, default=[])
                if cb:
                    results["country_breakdown"][term] = {r["country"]: r["count"] for r in cb}

            # NIOSH — Assay, SubjectOfInvestigation, EFO materials, measurements, ...
            rows = _safe(self.query_graph, "NIOSH", term)
            results["niosh_assays"].extend(rows)
            counts["niosh"] = _safe(self.count_graph, "NIOSH", term, default=0)
            if counts["niosh"]:
                type_counts["NIOSH"] = _safe(self.count_graph_by_label, "NIOSH", term, default=[])
                assay_names.update(_safe(self.distinct_niosh_labels, term))
                assay_names.update(_safe(self.distinct_graph_field, "NIOSH", term, "type"))
                niosh_mats.update(_safe(self.distinct_graph_field, "NIOSH", term, "C93410"))
                assay_uris.update(r["uri"] for r in rows if r.get("uri"))

            # NKB — Product, Assay, NPO_1680 measurements, media, publications, ...
            rows = _safe(self.query_graph, "NKB", term)
            results["nkb_materials"].extend(rows)
            counts["nkb"] = _safe(self.count_graph, "NKB", term, default=0)
            if counts["nkb"]:
                type_counts["NKB"] = _safe(self.count_graph_by_label, "NKB", term, default=[])
                nkb_mats.update(_safe(self.distinct_nkb_materials, term))
                nkb_meas.update(_safe(self.distinct_graph_field, "NKB", term, "C42614")[:40])

            results["match_counts"][term] = counts
            if type_counts:
                results["entity_type_counts"][term] = type_counts

        results["distinct_values"] = {
            "manufacturers": sorted(x for x in mfrs if x),
            "product_types": sorted(x for x in ptypes if x),
            "nanomaterials": sorted(x for x in nanos if x),
            "product_uris": sorted(cpsc_uris),
            "assay_names": sorted(x for x in assay_names if x),
            "assay_uris": sorted(assay_uris),
            "nkb_materials": sorted(x for x in nkb_mats if x),
            "niosh_materials": sorted(x for x in niosh_mats if x),
            "nkb_measurements": sorted(x for x in nkb_meas if x),
        }

        # ---- Failsafe: nothing matched -> broad retrieval so the LLM has context ----
        if (not results["cpsc_products"] and not results["niosh_assays"]
                and not results["nkb_materials"]):
            logger.info("no term matches; using broad failsafe retrieval")
            results["cpsc_products"] = _safe(self.query_graph, "CPSC", None)
            results["niosh_assays"] = _safe(self.query_graph, "NIOSH", None)
            results["nkb_materials"] = _safe(self.query_graph, "NKB", None)

        # ---- Nanomaterial distribution (always useful, e.g. 'most common') ----
        results["nanomaterial_stats"] = _safe(lambda: self.run_query("""
            MATCH (product)
            WHERE 'CPSC' IN product.graphs AND product.NPO_1808 IS NOT NULL
            WITH product.NPO_1808 AS material, count(*) AS count
            RETURN material, count ORDER BY count DESC
        """))

        try:
            if results["cpsc_products"] and results["niosh_assays"]:
                combined = []
                for product in results["cpsc_products"]:
                    nanomaterial = product.get("nanomaterial") or ""
                    if not isinstance(nanomaterial, str):
                        continue
                    for assay in results["niosh_assays"]:
                        desc = assay.get("description") or ""
                        if not isinstance(desc, str):
                            continue
                        if nanomaterial.lower() in desc.lower():
                            combined.append({"product": product, "assay": assay, "relevance": "Material match"})
                results["combined_data"] = combined
        except Exception as e:
            if self.verbose:
                print(f"Error finding connections: {e}")

        results["cpsc_products"] = self._deduplicate_by_uri(results["cpsc_products"])
        results["niosh_assays"] = self._deduplicate_by_uri(results["niosh_assays"])
        results["nkb_materials"] = self._deduplicate_by_uri(results["nkb_materials"])
        return results

    def format_results_for_context(self, results: Dict[str, Any]) -> str:
        """Format retrieval results into text context for the LLM."""
        context = "# Nanotoxicology Knowledge Base Results\n\n"

        # Exact match counts first (authoritative for "how many" questions).
        # Only surface terms that actually matched something, so stray 0-count
        # terms (from noisy LLM keyword extraction) cannot mislead the answer.
        match_counts = results.get("match_counts") or {}
        shown_counts = {t: c for t, c in match_counts.items()
                        if (c.get("cpsc") or c.get("cpsc_products") or 0) > 0
                        or (c.get("niosh") or c.get("niosh_assays") or 0) > 0
                        or (c.get("nkb") or c.get("nkb_matches") or 0) > 0}
        if shown_counts:
            context += "## Exact Match Counts (authoritative; use these for counting questions)\n\n"
            for term, c in shown_counts.items():
                parts = []
                n_cpsc = c.get("cpsc") or c.get("cpsc_products") or 0
                n_nkb = c.get("nkb") or c.get("nkb_matches") or 0
                n_niosh = c.get("niosh") or c.get("niosh_assays") or 0
                if n_cpsc:
                    parts.append(f"{n_cpsc} CPSC records")
                if n_nkb:
                    parts.append(f"{n_nkb} NKB records")
                if n_niosh:
                    parts.append(f"{n_niosh} NIOSH records")
                context += f"- '{term}': " + ", ".join(parts) + " (all entity types, case-insensitive)\n"
            context += "\n"

        type_counts = results.get("entity_type_counts") or {}
        if type_counts:
            context += "## Matching entity types by graph\n\n"
            for term, by_graph in type_counts.items():
                bits = []
                for g, rows in by_graph.items():
                    shown = ", ".join(f"{r['entity_type']}={r['count']}" for r in (rows or [])[:6])
                    if shown:
                        bits.append(f"{g}: {shown}")
                if bits:
                    context += f"- '{term}': " + " | ".join(bits) + "\n"
            context += "\n"

        # Full distinct-value lists (complete, not capped) for list questions.
        dv = results.get("distinct_values") or {}
        def _list_section(title, key, cap=80):
            vals = dv.get(key) or []
            if not vals:
                return ""
            shown = vals[:cap]
            s = f"## {title} ({len(vals)} total)\n\n"
            s += ", ".join(str(v) for v in shown)
            if len(vals) > cap:
                s += f", ... (+{len(vals) - cap} more)"
            return s + "\n\n"
        context += _list_section("Manufacturers of matching products", "manufacturers")
        context += _list_section("Matching CPSC nanomaterial labels", "nanomaterials")
        context += _list_section("Matching NKB material/knowledge labels", "nkb_materials")
        context += _list_section("Matching NKB measurement/parameter names", "nkb_measurements")
        context += _list_section("Matching NIOSH material IDs", "niosh_materials")
        context += _list_section("Matching assay / endpoint labels", "assay_names")

        # Country distribution (for 'products from <country>' style questions).
        cb = results.get("country_breakdown") or {}
        if cb:
            context += "## Country distribution of matching products (use for country-specific questions)\n\n"
            for term, dist in cb.items():
                pairs = ", ".join(f"{c}: {n}" for c, n in list(dist.items())[:25])
                context += f"- '{term}' by country: {pairs}\n"
            context += "\n"

        if results.get("cpsc_products"):
            context += "## CPSC records (sample)\n\n"
            for i, product in enumerate(results["cpsc_products"][:10], 1):
                labs = [l for l in (product.get("node_labels") or []) if l != "Resource"]
                context += f"{i}. {product.get('name') or product.get('product_type') or 'Record'}"
                if labs:
                    context += f" [{'/'.join(labs)}]"
                context += "\n"
                context += f"   Nanomaterial: {product.get('nanomaterial', 'Not specified')}\n"
                context += f"   Product type: {product.get('product_type', 'Not specified')}\n"
                context += f"   Manufacturer: {product.get('manufacturer', 'Not specified')}\n"
                context += f"   Country: {product.get('country', 'Not specified')}\n"
                context += f"   Category: {product.get('category', 'Not specified')}\n\n"

        if results.get("nanomaterial_stats"):
            stats = results["nanomaterial_stats"]
            # Explicit global ranking answer for "most common" questions,
            # excluding non-specific labels (Unknown/Unspecified/None/empty).
            _skip = {"unknown", "unspecified", "none", "n/a", "not specified", ""}
            specified = [s for s in stats
                         if str(s.get("material", "")).strip().lower() not in _skip]
            context += "## Nanomaterial Frequency Ranking (global, across all CPSC products)\n\n"
            if specified:
                top = specified[0]
                context += (f"The MOST COMMON specified nanomaterial overall is "
                            f"**{top['material']}** ({top['count']} products).\n\n")
            for i, stat in enumerate(specified[:8], 1):
                context += f"{i}. {stat['material']}: {stat['count']} products\n"
            context += "\n"

        if results.get("niosh_assays"):
            context += "## NIOSH records (sample; all entity types)\n\n"
            for i, assay in enumerate(results["niosh_assays"][:10], 1):
                labs = [l for l in (assay.get("node_labels") or []) if l != "Resource"]
                context += f"{i}. {assay.get('name') or assay.get('type') or assay.get('material_id') or 'Record'}"
                if labs:
                    context += f" [{'/'.join(labs)}]"
                context += "\n"
                if assay.get("description"):
                    context += f"   Description: {assay.get('description')}\n"
                if assay.get("type"):
                    context += f"   Type: {assay.get('type')}\n"
                if assay.get("material_id"):
                    context += f"   Material ID: {assay.get('material_id')}\n"
                if assay.get("value"):
                    unit = f" {assay.get('unit')}" if assay.get("unit") else ""
                    context += f"   Value: {assay.get('value')}{unit}\n"
                context += "\n"

        if results.get("nkb_materials"):
            context += "## NKB records (sample; all entity types)\n\n"
            for i, rec in enumerate(results["nkb_materials"][:10], 1):
                labs = [l for l in (rec.get("node_labels") or []) if l != "Resource"]
                context += f"{i}. {rec.get('name') or rec.get('nanomaterial') or rec.get('measurement') or 'Record'}"
                if labs:
                    context += f" [{'/'.join(labs)}]"
                context += "\n"
                if rec.get("nanomaterial"):
                    context += f"   Nanomaterial: {rec.get('nanomaterial')}\n"
                if rec.get("manufacturer"):
                    context += f"   Manufacturer: {rec.get('manufacturer')}\n"
                if rec.get("measurement"):
                    context += f"   Measurement: {rec.get('measurement')}\n"
                if rec.get("assay_type"):
                    context += f"   Assay type: {rec.get('assay_type')}\n"
                if rec.get("medium"):
                    context += f"   Medium: {str(rec.get('medium'))[:160]}\n"
                context += "\n"

        if results.get("combined_data"):
            context += "## Product-Assay Relationships\n\n"
            for i, item in enumerate(results["combined_data"][:5], 1):
                product = item["product"]
                assay = item["assay"]
                context += f"{i}. Product with {product.get('nanomaterial')} ({product.get('product_type', 'unknown')})\n"
                context += f"   Related assay: {assay.get('name', 'Unnamed')} - {assay.get('description', 'No description')}\n\n"

        return context

    # Shared prompt so answer_question and answer_question_traced stay in sync.
    _SYSTEM_PROMPT = (
        "You are an expert in nanotoxicology and nanomaterials science "
        "specializing in safety assessment and regulatory analysis."
    )

    def _build_answer_prompt(self, question: str, context: str) -> str:
        return f"""Use the following information from a nanotoxicology knowledge database to answer the question.

The context below contains AUTHORITATIVE, precomputed facts retrieved directly
from the database. Trust and use these numbers exactly. Specifically:
- "Exact Match Counts" gives the true number of matching records in each
  source graph (CPSC, NKB, NIOSH). For a "how many ..." question, state
  those exact numbers as the answer. Do not collapse graphs unless asked.
- "Country distribution of matching products" gives per-country counts. For an
  existence question like "are there any products from <country> that contain
  <material>?", look up that country: if its count is greater than 0, answer
  YES and give the count; if the country is absent or 0, answer NO.
- Distinct-value lists (manufacturers, nanomaterials, NKB labels,
  NIOSH material IDs, assay endpoints) are COMPLETE for the matched set;
  use them to enumerate examples.
- "Matching entity types by graph" says which node types hit (Product,
  Assay, measurement records, etc.). Use those counts; do not invent types.

Grounding rules (important):
- Base every factual claim on the context above. Do NOT invent numbers, product
  names, manufacturers, or countries that are not present in the context.
- Do NOT claim you lack information when the relevant count or value IS present.
- If the specific fact needed is genuinely absent from the context, say plainly
  that the knowledge base does not contain that information, and do not guess or
  substitute general/outside knowledge as if it were from the database.

KNOWLEDGE DATABASE CONTEXT:
{context}

QUESTION: {question}

Provide a clear answer that:
1. Directly answers the question first (the exact number, YES/NO, or list)
2. Cites specific examples/counts from the context
3. Explains the significance briefly in plain language
4. Notes data limitations only if genuinely relevant
"""

    def answer_question(self, question: str) -> str:
        """Run full RAG: analyze question -> retrieve -> format context -> generate answer."""
        analysis = self.analyze_question(question)
        if self.verbose:
            print(f"Question analysis: {analysis}")

        results = self.retrieve_relevant_data(analysis)
        if self.verbose:
            print(f"Retrieved CPSC={len(results['cpsc_products'])} "
                  f"NIOSH={len(results['niosh_assays'])} "
                  f"NKB={len(results['nkb_materials'])}")

        context = self.format_results_for_context(results)
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": self._build_answer_prompt(question, context)},
        ]
        answer = self.llm.generate(messages, max_new_tokens=1024, temperature=0.1)
        return answer.strip()

    def answer_question_traced(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline that returns the answer plus a structured trace for
        evaluation/logging: analysis, per-stage latency, retrieved entity sets,
        Cypher trace, and the final answer.
        """
        self.query_trace = []
        trace: Dict[str, Any] = {"question": question}

        t0 = time.perf_counter()
        analysis = self.analyze_question(question)
        analyze_ms = (time.perf_counter() - t0) * 1000.0
        trace["analysis"] = analysis

        t1 = time.perf_counter()
        results = self.retrieve_relevant_data(analysis)
        retrieve_ms = (time.perf_counter() - t1) * 1000.0

        # Extract retrieved entity sets (for retrieval precision/recall vs gold).
        # Prefer the full DISTINCT sets (not LIMIT-capped) when available.
        products = results.get("cpsc_products", [])
        assays = results.get("niosh_assays", [])
        dv = results.get("distinct_values", {}) or {}

        def _pref(distinct_key, row_vals):
            return dv[distinct_key] if dv.get(distinct_key) else sorted(set(row_vals))

        trace["retrieved"] = {
            "product_uris": _pref("product_uris", [p.get("uri") for p in products if p.get("uri")]),
            "nanomaterials": _pref("nanomaterials", [str(p.get("nanomaterial")) for p in products if p.get("nanomaterial")]),
            "manufacturers": _pref("manufacturers", [str(p.get("manufacturer")) for p in products if p.get("manufacturer")]),
            "product_types": _pref("product_types", [str(p.get("product_type")) for p in products if p.get("product_type")]),
            "assay_uris": _pref("assay_uris", [a.get("uri") for a in assays if a.get("uri")]),
            "assay_names": _pref("assay_names", [str(a.get("name")) for a in assays if a.get("name")]),
            "nkb_materials": dv.get("nkb_materials", []),
            "niosh_materials": dv.get("niosh_materials", []),
            "n_products": len(products),
            "n_assays": len(assays),
            "n_nkb": len(results.get("nkb_materials", [])),
            "match_counts": results.get("match_counts", {}),
            "entity_type_counts": results.get("entity_type_counts", {}),
        }

        context = self.format_results_for_context(results)
        user_prompt = self._build_answer_prompt(question, context)
        t2 = time.perf_counter()
        answer = self.llm.generate(
            [{"role": "system", "content": self._SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}],
            max_new_tokens=1024, temperature=0.1,
        )
        generate_ms = (time.perf_counter() - t2) * 1000.0

        trace["answer"] = answer.strip()
        trace["context_chars"] = len(context)
        trace["timings_ms"] = {
            "analyze": round(analyze_ms, 1),
            "retrieve": round(retrieve_ms, 1),
            "generate": round(generate_ms, 1),
            "total": round(analyze_ms + retrieve_ms + generate_ms, 1),
        }
        trace["cypher_trace"] = list(self.query_trace)
        logger.info("answered question=%r products=%d assays=%d total_ms=%.0f",
                    question, len(products), len(assays), trace["timings_ms"]["total"])
        return trace


def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _keyword_fallback_analysis(question: str) -> Dict[str, List[str]]:
    """Simple keyword extraction when LLM JSON fails."""
    q = question.lower()
    # Common nanomaterial and product/assay keywords
    keywords = []
    for word in re.findall(r"[a-z][a-z0-9]+", q):
        if len(word) > 2 and word not in ("the", "and", "for", "what", "about", "which", "that", "this", "from", "with", "have", "known", "their", "does", "there"):
                keywords.append(word)
    return {
        "nanomaterials": keywords[:5],
        "products": [],
        "assays": keywords[:5],
        "exposure_routes": [],
        "targets": [],
    }
