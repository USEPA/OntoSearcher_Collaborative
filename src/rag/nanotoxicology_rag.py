"""
Nanotoxicology RAG: same retrieval logic as llmexperiment.ipynb, with a pluggable LLM backend.
Supports OpenAI or a local model (e.g. Llama 2 7B) via the transformers library.
"""

import json
import re
from typing import List, Dict, Any, Optional

from .llm_backends import LLMBackend


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
        with self.driver.session() as session:
            results = session.run(query, **params).data()
            if self.verbose:
                print(f"\nQuery returned {len(results)} results")
            return results

    # ---------- Query helpers (same as notebook) ----------

    def query_cpsc_products(self, search_term: Optional[str] = None, limit: int = 50) -> List[Dict]:
        query = """
        MATCH (product)
        WHERE 'CPSC' IN product.graphs
        AND (
          $search_term IS NULL OR
          (product.NPO_1808 IS NOT NULL AND product.NPO_1808 CONTAINS $search_term) OR
          (product.C93401 IS NOT NULL AND product.C93401 CONTAINS $search_term) OR
          (product.C43530 IS NOT NULL AND product.C43530 CONTAINS $search_term)
        )
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

    def query_niosh_assays(self, search_term: Optional[str] = None, limit: int = 50) -> List[Dict]:
        query = """
        MATCH (assay)
        WHERE 'NIOSH' IN assay.graphs
        AND 'Assay' IN labels(assay)
        AND (
          $search_term IS NULL OR
          (assay.label IS NOT NULL AND assay.label CONTAINS $search_term) OR
          (assay.description IS NOT NULL AND assay.description CONTAINS $search_term)
        )
        RETURN
          assay.uri AS uri,
          assay.label AS name,
          assay.description AS description,
          assay.SIO_000300 AS value,
          assay.SIO_000221 AS unit
        LIMIT $limit
        """
        return self.run_query(query, {"search_term": search_term, "limit": limit})

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

    def retrieve_relevant_data(self, analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """Retrieve Neo4j data based on analysis (same logic as notebook)."""
        results = {
            "cpsc_products": [],
            "niosh_assays": [],
            "combined_data": [],
            "nanomaterial_stats": [],
        }

        try:
            for material in analysis.get("nanomaterials", []):
                results["cpsc_products"].extend(self.query_cpsc_products(material))
            for product in analysis.get("products", []):
                results["cpsc_products"].extend(self.query_cpsc_products(product))
        except Exception as e:
            if self.verbose:
                print(f"Error querying CPSC products: {e}")
            try:
                results["cpsc_products"] = self.run_query("""
                    MATCH (product)
                    WHERE 'CPSC' IN product.graphs AND product.NPO_1808 IS STRING
                    RETURN product.uri AS uri, product.NPO_1808 AS nanomaterial,
                           product.C43530 AS manufacturer, product.C93401 AS product_type,
                           product.C25464 AS country, product.C93400 AS category
                    LIMIT 10
                """)
            except Exception:
                pass

        try:
            for assay in analysis.get("assays", []):
                results["niosh_assays"].extend(self.query_niosh_assays(assay))
            for target in analysis.get("targets", []):
                results["niosh_assays"].extend(self.query_niosh_assays(target))
        except Exception as e:
            if self.verbose:
                print(f"Error querying NIOSH assays: {e}")
            try:
                results["niosh_assays"] = self.run_query("""
                    MATCH (assay)
                    WHERE 'NIOSH' IN assay.graphs AND 'Assay' IN labels(assay) AND assay.label IS STRING
                    RETURN assay.uri AS uri, assay.label AS name, assay.description AS description,
                           assay.SIO_000300 AS value, assay.SIO_000221 AS unit
                    LIMIT 10
                """)
            except Exception:
                pass

        try:
            results["nanomaterial_stats"] = self.run_query("""
                MATCH (product)
                WHERE 'CPSC' IN product.graphs AND product.NPO_1808 IS STRING
                WITH product.NPO_1808 AS material, count(*) AS count
                RETURN material, count ORDER BY count DESC
            """)
        except Exception as e:
            if self.verbose:
                print(f"Error getting nanomaterial stats: {e}")

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

        if results.get("cpsc_products"):
            context += "## Consumer Products containing Nanomaterials\n\n"
            for i, product in enumerate(results["cpsc_products"][:10], 1):
                context += f"{i}. Product: {product.get('product_type', 'Unknown')}\n"
                context += f"   Nanomaterial: {product.get('nanomaterial', 'Not specified')}\n"
                context += f"   Manufacturer: {product.get('manufacturer', 'Not specified')}\n"
                context += f"   Country: {product.get('country', 'Not specified')}\n"
                context += f"   Category: {product.get('category', 'Not specified')}\n\n"

        if results.get("nanomaterial_stats"):
            context += "## Nanomaterial Statistics\n\n"
            for i, stat in enumerate(results["nanomaterial_stats"][:5], 1):
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

    def answer_question(self, question: str) -> str:
        """Run full RAG: analyze question -> retrieve -> format context -> generate answer."""
        analysis = self.analyze_question(question)
        if self.verbose:
            print(f"Question analysis: {analysis}")

        results = self.retrieve_relevant_data(analysis)
        if self.verbose:
            print(f"Retrieved {len(results['cpsc_products'])} products and {len(results['niosh_assays'])} assays")

        context = self.format_results_for_context(results)

        system_prompt = (
            "You are an expert in nanotoxicology and nanomaterials science "
            "specializing in safety assessment and regulatory analysis."
        )
        user_prompt = f"""Use the following information from a nanotoxicology knowledge database to answer the question.
If the information needed is not available in the provided context, say that you don't have enough information from the knowledge base, but provide general information about the topic based on your knowledge.

KNOWLEDGE DATABASE CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer that:
1. Directly addresses the question
2. Cites specific examples from the knowledge base where relevant
3. Explains the significance of findings in plain language
4. Acknowledges data limitations where appropriate
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        answer = self.llm.generate(messages, max_new_tokens=1024, temperature=0.4)
        return answer.strip()


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
