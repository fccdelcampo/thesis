"""
Grounded KBPedia Retriever — Neo4j Edition
Queries KBPedia concepts stored in Neo4j (:KBPediaConcept nodes).

Two modes (controlled by `triple_filter` at construction time):
  triple_filter=False (default, FAST):
    - Entity extraction: keyword-based, zero LLM calls
    - Concept selection:  ONE LLM call at the end to filter all candidates at once
  triple_filter=True (RIGOROUS, highest quality):
    - Entity extraction: LLM extracts entities + MCQ choices
    - Triple selection:  LLM keeps only query-relevant triples per concept (N calls)
"""

import json
import re
from typing import List, Dict, Any, Optional


def _safe_parse_json(text: str, expect: type = None):
    """Robustly extract and parse the first complete JSON value from LLM output.
    - Strips markdown fences.
    - Finds the first '{' (for dict) or '[' (for list) and stops after the matching close.
    - Uses raw_decode so trailing text/extra objects never cause 'Extra data' errors.
    - Returns None on empty input or parse failure instead of raising."""
    if not text or not text.strip():
        return None
    # Strip markdown fences
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0].strip()
    if not t:
        return None
    # Find the first opening char based on expected type
    opener = '{' if expect is dict else ('[' if expect is list else None)
    if opener is None:
        # Auto-detect: whichever comes first
        i_brace = t.find('{')
        i_bracket = t.find('[')
        if i_brace == -1 and i_bracket == -1:
            return None
        if i_brace == -1:
            pos = i_bracket
        elif i_bracket == -1:
            pos = i_brace
        else:
            pos = min(i_brace, i_bracket)
    else:
        pos = t.find(opener)
        if pos == -1:
            return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t, pos)
        return obj
    except (json.JSONDecodeError, ValueError):
        return None


class KBPediaRetriever:
    """
    Retrieves grounded knowledge from KBPedia concepts stored in Neo4j.
    Fast mode (triple_filter=False, default): keyword entity extraction + single end-of-search LLM filter.
    Rigorous mode (triple_filter=True): LLM entity extraction + per-concept LLM triple filtering.
    """

    _index_verified = False

    def __init__(self, driver, llm=None, top_k: int = 4, embedder=None):
        self.driver  = driver
        self.embedder = embedder
        
        try:
            from graphrag_pipeline.graphrag.config import ENABLE_LLM_FILTERING
        except ModuleNotFoundError:
            try:
                from graphrag.config import ENABLE_LLM_FILTERING
            except ImportError:
                ENABLE_LLM_FILTERING = True

        # Nullify LLM if the global config toggle is disabled
        self.llm = llm if ENABLE_LLM_FILTERING else None
        
        self.top_k   = top_k

    def _verify_index(self):
        """Check once whether the KBPedia vector index exists."""
        if KBPediaRetriever._index_verified:
            return True
        try:
            records, _, _ = self.driver.execute_query(
                "SHOW VECTOR INDEXES YIELD name WHERE name = 'kbpediaConceptVectorIndex' RETURN name",
                database_="neo4j",
            )
            if records:
                KBPediaRetriever._index_verified = True
                return True
            else:
                print("DEBUG: KBPedia vector index 'kbpediaConceptVectorIndex' not found. "
                      "Run kbpedia_loader.py first to embed KBPedia concepts.")
                return False
        except Exception as e:
            print(f"DEBUG: Error verifying KBPedia vector index: {e}")
            return False

    def _vector_search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search against KBPedia concepts."""
        if not query_vector:
            return []

        try:
            records, _, _ = self.driver.execute_query(
                """
                CALL db.index.vector.queryNodes('kbpediaConceptVectorIndex', $limit, $vector)
                YIELD node, score
                RETURN node.uri AS uri, node.name AS name, node.definition AS definition,
                       node.altLabels AS altLabels, score
                """,
                vector=query_vector,
                limit=limit,
                database_="neo4j",
            )
            results = []
            for r in records:
                results.append({
                    "uri": r["uri"],
                    "name": r["name"],
                    "definition": r["definition"] or "",
                    "altLabels": r["altLabels"] or [],
                    "score": r["score"],
                })
            return results
        except Exception as e:
            print(f"DEBUG: KBPedia concept vector search error: {e}")
            return []



    def get_neighborhood(self, uri: str) -> List[str]:
        """Get the ontology neighborhood of a concept (ancestors, children, siblings)."""
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (n:KBPediaConcept {uri: $uri})
                
                CALL (n) {
                    OPTIONAL MATCH (n)-[:SUBCLASS_OF*1..2]->(ancestor:KBPediaConcept)
                    RETURN collect(DISTINCT {name: ancestor.name, def: ancestor.definition})[0..8] AS ancestors
                }
                
                CALL (n) {
                    OPTIONAL MATCH (child:KBPediaConcept)-[:SUBCLASS_OF]->(n)
                    RETURN collect(DISTINCT {name: child.name, def: child.definition})[0..12] AS children
                }
                
                CALL (n) {
                    OPTIONAL MATCH (n)-[:SUBCLASS_OF]->(:KBPediaConcept)<-[:SUBCLASS_OF]-(sibling:KBPediaConcept)
                    WHERE sibling.uri <> n.uri
                    RETURN collect(DISTINCT {name: sibling.name, def: sibling.definition})[0..5] AS siblings
                }
                
                RETURN ancestors, children, siblings
                """,
                uri=uri,
                database_="neo4j",
            )

            triples = []
            if records:
                row = records[0]
                for a in row.get("ancestors", []):
                    if a and a.get("name"):
                        # Only keep the ancestor NAME — ancestor definition blurbs are long
                        # encyclopedia text that bloats context without aiding the LLM.
                        triples.append(f"subclass of: {a['name']}")
                for c in row.get("children", []):
                    if c and c.get("name"):
                        triples.append(f"has subclass: {c['name']}")
                for s in row.get("siblings", []):
                    if s and s.get("name"):
                        triples.append(f"sibling concept: {s['name']}")
            return triples
        except Exception as e:
            print(f"DEBUG: KBPedia neighborhood error: {e}")
            return []

    def filter_triples_for_query(self, query: str, entity_label: str, triples: List[str], original_query: str = "") -> List[str]:
        """Use LLM to keep only logically relevant triples."""
        if not self.llm or not triples:
            return triples[:15]

        # Use the full original prompt (with MCQ choices) when available for better filtering
        filter_query = original_query if original_query else query

        triples_text = "\n".join([f"- {t}" for t in triples])
        prompt = f"""Task: You are deeply integrated into a Prolog GraphRAG pipeline.
Below is a list of known facts (triples) about the entity '{entity_label}'.
Your job is to strictly filter these facts and select ONLY the ones that are logically relevant or useful to answer the user's query.
Irrelevant facts will poison the logic engine, so drop anything that doesn't help.

User Query: "{filter_query}"

Facts about {entity_label}:
{triples_text}

Output format: A JSON list of strings containing exactly the original facts you deemed relevant.
Example: ["subclass of: Structural Component", "ancestor definition: provides support"]
If none are relevant, output an empty list [].
"""
        try:
            raw = self.llm.invoke(prompt)
            # .content may exist (LLM response object) or it may already be a plain string
            res = (raw.content if hasattr(raw, 'content') else str(raw)).strip()
            print(f"DEBUG: Triple filter raw response for '{entity_label}': {repr(res[:200])}", flush=True)

            # Strip markdown code fences
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()

            # Guard: DebugOllamaLLM._clean_response_text() strips everything outside {},
            # so a JSON array comes back empty. Try to find brackets first.
            bracket_start = res.find('[')
            bracket_end = res.rfind(']')
            if bracket_start != -1 and bracket_end != -1:
                res = res[bracket_start:bracket_end + 1]
            elif not res:
                print(f"DEBUG: Triple filter got empty response for '{entity_label}'. Using all raw triples.", flush=True)
                return triples

            filtered = json.loads(res)
            if isinstance(filtered, list):
                return filtered
        except Exception as e:
            print(f"DEBUG: Triple filtering failed for '{entity_label}': {e}. Using all raw triples.", flush=True)

        return triples

    def filter_triples_batch(self, query: str, concepts_data: List[Dict[str, Any]], original_query: str = "") -> Dict[str, List[str]]:
        """
        Use LLM to keep only logically relevant triples for MULTIPLE concepts at once.
        concepts_data is a list of dicts: {"name": str, "triples": List[str]}
        Returns a dictionary mapping concept_name to a list of filtered triples.
        """
        if not self.llm or not concepts_data:
            return {c["name"]: c["triples"][:15] for c in concepts_data}

        filter_query = original_query if original_query else query

        # Build the payload for the prompt
        # Cap triples per concept to 12 to prevent the prompt from exceeding num_ctx.
        # With 5 concepts * ~25 triples, the prompt alone can blow past 2048 tokens
        # causing Ollama to truncate context and hang indefinitely on malformed JSON.
        MAX_TRIPLES_PER_CONCEPT = 12
        concepts_text = ""
        for c in concepts_data:
            capped_triples = c["triples"][:MAX_TRIPLES_PER_CONCEPT]
            triples_text = "\n".join([f"  - {t}" for t in capped_triples])
            concepts_text += f"Concept: '{c['name']}'\nFacts:\n{triples_text}\n\n"

        prompt = f"""Task: You are a strict knowledge filter for a Prolog GraphRAG pipeline.
Below are ontology facts for multiple concepts retrieved from KBPedia.
The user is answering a specific question — possibly a multiple-choice question (MCQ).

User Query: "{filter_query}"

{concepts_text}

STRICT FILTERING RULES:
1. Keep a fact ONLY if it directly helps determine which MCQ answer choice is correct or incorrect.
   - A 'subclass of' or 'has subclass' fact is useful ONLY if the parent/child class name directly
     relates to the mechanism described in the question (e.g. 'subclass of: plate_convergence' for a
     tectonic question, NOT 'subclass of: scalar quantity' for a geography question).
   - Sibling concepts are almost never useful — keep one ONLY if it is literally one of the MCQ choices.
2. DROP all of: unrelated subclasses, unrelated siblings, unrelated parent classes, etymology,
   Wikipedia metadata, and any fact whose connection to the question you cannot explain in one sentence.
3. For concepts that ARE an MCQ answer choice, keep its core definition PLUS any fact that explains
   how/why it forms or doesn't form under the conditions described in the question.
4. If a concept has NO relevant facts after filtering, output an empty list [] for it.
   DO NOT force-keep facts just to have something.

Output format: A JSON dictionary where keys are the Concept names, and values are lists of kept fact strings.
Example:
{{
  "trench": ["definition: a long narrow excavation in the ground", "subclass of: topographical feature"],
  "temperature": []
}}
"""
        # Default fallback
        result_map = {c["name"]: c["triples"] for c in concepts_data}

        try:
            raw = self.llm.invoke(prompt)
            res = (raw.content if hasattr(raw, 'content') else str(raw)).strip()
            safe_res = str(res[:200]).encode('ascii', 'replace').decode('ascii')
            print(f"DEBUG: Batch triple filter raw response: {repr(safe_res)}...", flush=True)

            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()

            bracket_start = res.find('{')
            bracket_end = res.rfind('}')
            if bracket_start != -1 and bracket_end != -1:
                res = res[bracket_start:bracket_end + 1]
            elif not res:
                print(f"DEBUG: Batch triple filter got empty response. Using all raw triples.", flush=True)
                return result_map

            filtered = _safe_parse_json(res, expect=dict)
            if isinstance(filtered, dict):
                for c in concepts_data:
                    name = c["name"]
                    if name in filtered and isinstance(filtered[name], list):
                        # Clean up formatting artifacts (like "- subclass of:" -> "subclass of:")
                        cleaned_triples = [t[2:].strip() if t.startswith("- ") else t.strip() for t in filtered[name]]
                        result_map[name] = cleaned_triples
        except Exception as e:
            print(f"DEBUG: Batch triple filtering failed: {e}. Using all raw triples.", flush=True)

        return result_map

    def _filter_concepts_once(self, query_text: str, candidates: list, original_query: str = "") -> list:
        """
        Single LLM call to filter all candidates at once.
        `candidates` is a list of dicts: {name, uri, definition, triples}.
        `original_query` is the full user prompt including MCQ choices (when available).
        Returns a filtered list of dicts containing only relevant candidates.
        """
        if not self.llm or not candidates:
            return candidates

        # Use the full original prompt (with MCQ choices) when available for better filtering
        filter_query = original_query if original_query else query_text

        # Extract MCQ choices from the question so the LLM knows to keep them
        mcq_choices = re.findall(r'\b[A-D][.)]\s*(.+)', filter_query)
        mcq_hint = ""
        if mcq_choices:
            mcq_hint = (
                f"\n\n### MCQ Answer Choices Detected\n"
                f"The question has these answer options: {mcq_choices}\n"
                f"You MUST keep any concept whose name or definition relates to ANY of these choices, even loosely. "
                f"These are exactly the concepts the Prolog engine needs to reason about.\n"
            )

        summary_lines = []
        for idx, c in enumerate(candidates):
            defn = (c.get("definition") or "")[:400]
            summary_lines.append(f"[{idx}] {c['name']}: {defn}")
        summary_text = "\n".join(summary_lines)

        prompt = f"""You are filtering knowledge graph concepts for a Prolog-based question-answering system.

User question: "{filter_query}"{mcq_hint}

Candidates (index: concept name: short definition):
{summary_text}

### Decision Criteria
Keep a concept if ANY of the following is true:
1. Its name or definition directly relates to the question topic.
2. Its name or definition relates to ANY of the MCQ answer choices listed above.
3. It provides a fact or rule that could help the Prolog engine derive an answer.

Reject a concept ONLY if it is clearly off-topic (e.g., unrelated domain, coincidental word match with no logical connection).
When in doubt, KEEP IT — missing a concept is worse than keeping an extra one.

Return a JSON list of the RELEVANT indices (integers). Example: [0, 2, 4, 7, 11]
Output ONLY the JSON list, nothing else.
"""
        try:
            raw = self.llm.invoke(prompt)
            res = (raw.content if hasattr(raw, 'content') else str(raw)).strip()
            if "```" in res:
                res = res.split("```")[1].split("```")[0].strip()
            bracket_start = res.find('[')
            bracket_end = res.rfind(']')
            if bracket_start != -1 and bracket_end != -1:
                res = res[bracket_start:bracket_end + 1]
            indices = _safe_parse_json(res, expect=list)
            if not isinstance(indices, list):
                raise ValueError(f"LLM did not return a list: {repr(res[:100])}")
            print(f"DEBUG: End-filter LLM kept indices: {indices} from {len(candidates)} candidates.", flush=True)
            return [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        except Exception as e:
            print(f"DEBUG: End-filter LLM call failed: {e}. Returning all candidates.", flush=True)
            return candidates

    def search(self, query_text: str, top_k: int = 5, original_query: str = "", seeded_entities: List[str] = None) -> Any:
        from .retriever import RetrieverResult, RetrieverResultItem

        if not self._verify_index():
            return RetrieverResult(items=[], metadata={})

        # `original_query` = full user prompt (with MCQ choices) for LLM filtering.
        filter_query = original_query if original_query else query_text

        # ── Vector Search ───────────────────────────────────────────────────
        print(f"DEBUG: [KBPedia] Computing query embedding to search KBPedia concepts...", flush=True)
        try:
            if self.embedder:
                query_vector = self.embedder.embed_query("query: " + query_text)
            else:
                from sentence_transformers import SentenceTransformer
                print("DEBUG: Pipeline embedder missing, falling back to local SentenceTransformer.", flush=True)
                temp_embedder = SentenceTransformer("all-MiniLM-L6-v2")
                query_vector = temp_embedder.encode(query_text).tolist()
                
            # Grab top_k * 2 candidates to feed into the LLM logic validator later
            matches = self._vector_search(query_vector, limit=top_k * 2)
        except Exception as e:
            print(f"DEBUG: KBPedia embedding/search failed: {e}", flush=True)
            matches = []

        seen_uris = set()
        candidates = []  # list of {name, uri, definition, triples}
        items = []       # final result items

        if not matches:
            print(f"DEBUG: KBPedia has no vector matches for '{query_text}'.", flush=True)
        else:
            print(f"DEBUG: KBPedia matched {len(matches)} concept(s) via vector similarity: {[m['name'] for m in matches]}", flush=True)

        for match in matches:
            uri = match["uri"]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            # Build raw triples
            raw_triples = []
            if match["definition"]:
                raw_triples.append(f"definition: {match['definition'][:400]}")
            raw_triples.extend(self.get_neighborhood(uri))

            # Defensive filter: remove any triple that looks like an MCQ option
            mcq_pattern = re.compile(r'^[A-D][.):] ', re.IGNORECASE)
            raw_triples = [t for t in raw_triples if not mcq_pattern.match(t)]

            if not raw_triples:
                print(f"DEBUG: No raw triples for '{match['name']}' ({uri}), skipping.", flush=True)
                continue

            # Buffer raw candidates for a single end-of-loop LLM pass
            print(f"DEBUG: Buffered {len(raw_triples)} raw triple(s) for '{match['name']}'", flush=True)
            candidates.append({
                "name": match["name"],
                "uri": uri,
                "definition": match.get("definition", ""),
                "triples": raw_triples,
            })

        # ── Single LLM relevance filter over all candidates ─────────────
        if candidates:
            filtered_candidates = self._filter_concepts_once(query_text, candidates, original_query=filter_query)
            print(f"DEBUG: End-filter kept {len(filtered_candidates)}/{len(candidates)} concepts.", flush=True)
            
            # Batch Triple filtering on the validated concepts
            filtered_concepts_to_use = filtered_candidates[:top_k]
            
            if filtered_concepts_to_use:
                # Prepare batch payload
                batch_data = [{"name": c["name"], "triples": c["triples"]} for c in filtered_concepts_to_use]
                print(f"DEBUG: Batch sending {len(batch_data)} concepts to LLM triple filter...", flush=True)
                
                # We can process all top_k (usually <= 5) in one LLM call
                batch_results = self.filter_triples_batch(query_text, batch_data, original_query=filter_query)
                
                # Reassign filtered triples back to the candidates
                for c in filtered_concepts_to_use:
                    selected = batch_results.get(c["name"], c["triples"])
                    print(f"DEBUG: Batch LLM filter kept {len(selected)}/{len(c['triples'])} triple(s) for '{c['name']}'", flush=True)
                    c["triples"] = selected

                # Remove concepts where LLM filter kept 0 triples (completely irrelevant)
                before_count = len(filtered_concepts_to_use)
                filtered_concepts_to_use = [c for c in filtered_concepts_to_use if c["triples"]]
                removed = before_count - len(filtered_concepts_to_use)
                if removed:
                    print(f"DEBUG: Removed {removed} concept(s) with 0 surviving triples.", flush=True)

            for c in filtered_concepts_to_use:
                selected_triples = c["triples"]

                triples_text = "\n".join([f"  - {t}" for t in selected_triples])
                content = f"KBPedia Concept: {c['name']}.\nRelevant Logical Facts:\n{triples_text}"
                items.append(RetrieverResultItem(
                    content=content,
                    metadata={"source": "KBPedia", "entity": c["name"],
                              "uri": c["uri"], "score": 1.0, "triples": selected_triples},
                ))

        print(f"DEBUG: KBPedia found {len(items)} grounded concept contexts.", flush=True)
        return RetrieverResult(items=items, metadata={})

