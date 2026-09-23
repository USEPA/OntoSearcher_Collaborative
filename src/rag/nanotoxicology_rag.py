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

    # ---------- Query helpers (same as notebook) ----------

    # Case-insensitive substring match; empty/very short terms are skipped by callers.
    _CPSC_WHERE = """
        WHERE 'CPSC' IN product.graphs
        AND (
          $search_term IS NULL OR $search_term = '' OR
          (product.NPO_1808 IS NOT NULL AND toLower(product.NPO_1808) CONTAINS toLower($search_term)) OR
          (product.C93401 IS NOT NULL AND toLower(product.C93401) CONTAINS toLower($search_term)) OR
          (product.C43530 IS NOT NULL AND toLower(product.C43530) CONTAINS toLower($search_term)) OR
          (product.C93400 IS NOT NULL AND toLower(product.C93400) CONTAINS toLower($search_term))
        )
    """

    def query_cpsc_products(self, search_term: Optional[str] = None, limit: int = 200) -> List[Dict]:
        query = f"""
        MATCH (product)
        {self._CPSC_WHERE}
        RETURN
          product.uri AS uri,
          product.NPO_1808 AS nanomaterial,
          product.C43530 AS manufacturer,
          product.C93401 AS product_type,
          product.C25464 AS country,
          product.C93400 AS category
        LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def count_cpsc_products(self, search_term: Optional[str] = None) -> int:
        """Total number of CPSC products matching the term (case-insensitive).
        Enables accurate answers to 'how many ...' questions (not capped by LIMIT).
        """
        query = f"MATCH (product) {self._CPSC_WHERE} RETURN count(product) AS c"
        rows = self.run_query(query, {"search_term": search_term})
        return int(rows[0]["c"]) if rows else 0

    _NIOSH_WHERE = """
        WHERE 'NIOSH' IN assay.graphs
        AND 'Assay' IN labels(assay)
        AND (
          $search_term IS NULL OR $search_term = '' OR
          (assay.label IS NOT NULL AND toLower(assay.label) CONTAINS toLower($search_term)) OR
          (assay.description IS NOT NULL AND toLower(assay.description) CONTAINS toLower($search_term))
        )
    """

    def query_niosh_assays(self, search_term: Optional[str] = None, limit: int = 200) -> List[Dict]:
        query = f"""
        MATCH (assay)
        {self._NIOSH_WHERE}
        RETURN
          assay.uri AS uri,
          assay.label AS name,
          assay.description AS description,
          assay.SIO_000300 AS value,
          assay.SIO_000221 AS unit
        LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def count_niosh_assays(self, search_term: Optional[str] = None) -> int:
        """Total number of NIOSH assays matching the term (case-insensitive)."""
        query = f"MATCH (assay) {self._NIOSH_WHERE} RETURN count(assay) AS c"
        rows = self.run_query(query, {"search_term": search_term})
        return int(rows[0]["c"]) if rows else 0

    # ---------- distinct-value helpers (accurate lists, not LIMIT-capped) ----------

    def distinct_cpsc_field(self, search_term: str, field: str) -> List[str]:
        """Distinct values of a CPSC property (e.g. C43530 manufacturers) for a term."""
        allowed = {"manufacturer": "C43530", "product_type": "C93401",
                   "category": "C93400", "nanomaterial": "NPO_1808", "country": "C25464"}
        prop = allowed.get(field, field)
        query = f"""
        MATCH (product)
        {self._CPSC_WHERE}
        AND product.{prop} IS NOT NULL
        RETURN DISTINCT product.{prop} AS value
        """
        return [r["value"] for r in self.run_query(query, {"search_term": search_term})]

    def cpsc_country_breakdown(self, search_term: str, limit: int = 25) -> List[Dict]:
        """Country distribution (C25464) among CPSC products matching a term.
        Enables conjunctive 'products from <country> containing <material>' answers.
        """
        query = f"""
        MATCH (product)
        {self._CPSC_WHERE}
        AND product.C25464 IS NOT NULL
        WITH product.C25464 AS country, count(*) AS count
        RETURN country, count ORDER BY count DESC LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

    def distinct_niosh_labels(self, search_term: str) -> List[str]:
        query = f"""
        MATCH (assay)
        {self._NIOSH_WHERE}
        AND assay.label IS NOT NULL
        RETURN DISTINCT assay.label AS value
        """
        return [r["value"] for r in self.run_query(query, {"search_term": search_term})]

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
        for grp in groups:
            for t in (grp or []):
                t = str(t).strip().strip(".,;:\"'")
                if len(t) < 2:
                    continue
                words = t.split()
                if len(words) > 3:
                    continue  # free-text phrase, not an entity keyword
                if all(w.lower() in cls._GENERIC_TOKENS for w in words):
                    continue  # nothing but generic filler
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(t)
        return out

    def retrieve_relevant_data(self, analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """Retrieve Neo4j data based on analysis.

        Robust search: case-insensitive matching, exact match COUNTS (for
        'how many' questions), full DISTINCT value lists (for list questions,
        not capped by LIMIT), and failsafes when the LLM analysis is empty or a
        term returns nothing.
        """
        results: Dict[str, Any] = {
            "cpsc_products": [], "niosh_assays": [], "combined_data": [],
            "nanomaterial_stats": [], "match_counts": {},
            "distinct_values": {"manufacturers": [], "product_types": [], "nanomaterials": [],
                                "product_uris": [], "assay_names": [], "assay_uris": []},
            "country_breakdown": {},
            "search_terms": {"cpsc": [], "niosh": []},
        }

        cpsc_terms = self._clean_terms(analysis.get("nanomaterials"), analysis.get("products"))
        niosh_terms = self._clean_terms(analysis.get("assays"), analysis.get("targets"))
        # Cross-store failsafe terms: also try the "other" bucket if a store gets nothing.
        all_terms = self._clean_terms(cpsc_terms, niosh_terms,
                                      analysis.get("exposure_routes"), analysis.get("targets"))
        results["search_terms"] = {"cpsc": cpsc_terms, "niosh": niosh_terms}

        def _safe(fn, *a, default=None):
            try:
                return fn(*a)
            except Exception as e:
                logger.warning("retrieval sub-query failed: %s", e)
                return default if default is not None else []

        # ---- CPSC products (case-insensitive), with counts + distinct values ----
        cpsc_uris, mfrs, ptypes, nanos = set(), set(), set(), set()
        for term in (cpsc_terms or all_terms):
            rows = _safe(self.query_cpsc_products, term)
            results["cpsc_products"].extend(rows)
            cnt = _safe(self.count_cpsc_products, term, default=0)
            results["match_counts"].setdefault(term, {})["cpsc_products"] = cnt
            if cnt:
                mfrs.update(_safe(self.distinct_cpsc_field, term, "manufacturer"))
                ptypes.update(_safe(self.distinct_cpsc_field, term, "product_type"))
                nanos.update(_safe(self.distinct_cpsc_field, term, "nanomaterial"))
                cpsc_uris.update(_safe(self.distinct_cpsc_field, term, "uri"))
                cb = _safe(self.cpsc_country_breakdown, term, default=[])
                if cb:
                    results["country_breakdown"][term] = {r["country"]: r["count"] for r in cb}

        # ---- NIOSH assays (case-insensitive), with counts + distinct labels ----
        assay_uris, assay_names = set(), set()
        for term in (niosh_terms or all_terms):
            rows = _safe(self.query_niosh_assays, term)
            results["niosh_assays"].extend(rows)
            cnt = _safe(self.count_niosh_assays, term, default=0)
            results["match_counts"].setdefault(term, {})["niosh_assays"] = cnt
            if cnt:
                assay_names.update(_safe(self.distinct_niosh_labels, term))
                assay_uris.update(r["uri"] for r in rows if r.get("uri"))

        results["distinct_values"] = {
            "manufacturers": sorted(x for x in mfrs if x),
            "product_types": sorted(x for x in ptypes if x),
            "nanomaterials": sorted(x for x in nanos if x),
            "product_uris": sorted(cpsc_uris),
            "assay_names": sorted(x for x in assay_names if x),
            "assay_uris": sorted(assay_uris),
        }

        # ---- Failsafe: nothing matched -> broad retrieval so the LLM has context ----
        if not results["cpsc_products"] and not results["niosh_assays"]:
            logger.info("no term matches; using broad failsafe retrieval")
            results["cpsc_products"] = _safe(self.query_cpsc_products, None)  # all (up to limit)
            results["niosh_assays"] = _safe(self.query_niosh_assays, None)

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
        return results

    def format_results_for_context(self, results: Dict[str, Any]) -> str:
        """Format retrieval results into text context for the LLM."""
        context = "# Nanotoxicology Knowledge Base Results\n\n"

        # Exact match counts first (authoritative for "how many" questions).
        # Only surface terms that actually matched something, so stray 0-count
        # terms (from noisy LLM keyword extraction) cannot mislead the answer.
        match_counts = results.get("match_counts") or {}
        shown_counts = {t: c for t, c in match_counts.items()
                        if (c.get("cpsc_products") or 0) > 0 or (c.get("niosh_assays") or 0) > 0}
        if shown_counts:
            context += "## Exact Match Counts (authoritative; use these for counting questions)\n\n"
            for term, c in shown_counts.items():
                parts = []
                if c.get("cpsc_products"):
                    parts.append(f"{c['cpsc_products']} CPSC products")
                if c.get("niosh_assays"):
                    parts.append(f"{c['niosh_assays']} NIOSH assays")
                context += f"- '{term}': " + ", ".join(parts) + " (case-insensitive match)\n"
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
        context += _list_section("Matching nanomaterial labels", "nanomaterials")
        context += _list_section("Matching assay endpoints", "assay_names")

        # Country distribution (for 'products from <country>' style questions).
        cb = results.get("country_breakdown") or {}
        if cb:
            context += "## Country distribution of matching products (use for country-specific questions)\n\n"
            for term, dist in cb.items():
                pairs = ", ".join(f"{c}: {n}" for c, n in list(dist.items())[:25])
                context += f"- '{term}' by country: {pairs}\n"
            context += "\n"

        if results.get("cpsc_products"):
            context += "## Consumer Products containing Nanomaterials\n\n"
            for i, product in enumerate(results["cpsc_products"][:10], 1):
                context += f"{i}. Product: {product.get('product_type', 'Unknown')}\n"
                context += f"   Nanomaterial: {product.get('nanomaterial', 'Not specified')}\n"
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
            context += "## Toxicology Assay Information\n\n"
            for i, assay in enumerate(results["niosh_assays"][:10], 1):
                context += f"{i}. Assay: {assay.get('name', 'Unknown')}\n"
                context += f"   Description: {assay.get('description', 'Not specified')}\n"
                if assay.get("value"):
                    context += f"   Value: {assay.get('value')}"
                    if assay.get("unit"):
                        context += f" {assay.get('unit')}"
                    context += "\n"
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
- "Exact Match Counts" gives the true number of matching records. For a
  "how many ..." question, state that exact number as the answer.
- "Country distribution of matching products" gives per-country counts. For an
  existence question like "are there any products from <country> that contain
  <material>?", look up that country: if its count is greater than 0, answer
  YES and give the count; if the country is absent or 0, answer NO.
- Distinct-value lists (manufacturers, nanomaterials, assay endpoints) are
  COMPLETE for the matched set; use them to enumerate examples.

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
            print(f"Retrieved {len(results['cpsc_products'])} products and {len(results['niosh_assays'])} assays")

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
            "n_products": len(products),
            "n_assays": len(assays),
            "match_counts": results.get("match_counts", {}),
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
