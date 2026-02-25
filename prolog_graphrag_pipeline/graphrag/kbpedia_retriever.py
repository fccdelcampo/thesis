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


class KBPediaRetriever:
    """
    Retrieves grounded knowledge from KBPedia concepts stored in Neo4j.
    Fast mode (triple_filter=False, default): keyword entity extraction + single end-of-search LLM filter.
    Rigorous mode (triple_filter=True): LLM entity extraction + per-concept LLM triple filtering.
    """

    _index_verified = False

    def __init__(self, driver, llm=None, top_k: int = 4, triple_filter: bool = False):
        self.driver  = driver
        self.llm     = llm
        self.top_k   = top_k
        self.triple_filter = triple_filter  # False = fast (1 LLM call at end); True = rigorous per-concept LLM filter

    def _verify_index(self):
        """Check once whether the KBPedia fulltext index exists."""
        if KBPediaRetriever._index_verified:
            return True
        try:
            records, _, _ = self.driver.execute_query(
                "SHOW FULLTEXT INDEXES YIELD name WHERE name = 'kbpediaConceptIndex' RETURN name",
                database_="neo4j",
            )
            if records:
                KBPediaRetriever._index_verified = True
                return True
            else:
                print("DEBUG: KBPedia fulltext index 'kbpediaConceptIndex' not found. "
                      "Run kbpedia_loader.py first to load KBPedia into Neo4j.")
                return False
        except Exception as e:
            print(f"DEBUG: Error verifying KBPedia index: {e}")
            return False

    def _extract_entities_fast(self, query: str) -> List[str]:
        """
        Fast keyword-based entity extraction — no LLM calls.
        Strips MCQ option markers (A. B. C. D.), removes stop words,
        and returns the top N content words sorted by length (most specific first).
        """
        stop_words = {
            'what', 'which', 'where', 'when', 'who', 'how', 'the', 'that',
            'this', 'with', 'from', 'into', 'have', 'most', 'some', 'such',
            'does', 'will', 'each', 'they', 'their', 'would', 'could', 'more',
            'than', 'also', 'like', 'make', 'used', 'being', 'been', 'were',
            'about', 'other', 'only', 'over', 'both', 'very', 'after',
        }
        # Remove MCQ option labels and punctuation
        cleaned = re.sub(r'\b[A-D][.)]\s?', ' ', query)
        cleaned = re.sub(r'[?!,;:()\[\]]', ' ', cleaned).replace('\n', ' ')
        words = cleaned.split()
        # Keep words > 4 chars not in stop list
        content = [
            w.strip('.') for w in words
            if len(w) > 4 and w.lower().strip('.') not in stop_words
        ]
        # Deduplicate (case-insensitive), longest first
        seen: set = set()
        unique = []
        for w in sorted(content, key=len, reverse=True):
            key = w.lower()
            if key not in seen:
                seen.add(key)
                unique.append(w)
        return unique[:8]

    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        LLM-assisted entity extraction (used only when use_llm=True).
        Extracts explicit entities, functional/conceptual terms, AND MCQ choices.
        Falls back to simple word-splitting if the LLM is unavailable.
        """
        if not self.llm:
            return [w for w in query.replace('?', '').split() if len(w) > 4][:5]

        prompt = f"""You are extracting search terms for a knowledge base lookup.

Given this query, extract ALL of the following:
1. **Explicit entities** mentioned (e.g., "plant cell", "animal skeleton")
2. **Implied functional concepts** that the question is really asking about (e.g., if the question compares functions, extract the function like "structural support", "protection")
3. **All MCQ answer choices** if they exist (e.g., "cell membrane", "cell wall", "chloroplast", "nucleus")
4. **Related scientific terms** that would help answer (e.g., "rigidity", "endoskeleton")

Query: "{query}"

Output a plain JSON list of strings. Include 5-8 terms maximum. Do not include anything else.
Example: ["plant cell", "animal skeleton", "structural support", "cell wall", "cell membrane", "chloroplast", "nucleus"]"""
        try:
            response = self.llm.invoke(prompt).content.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Guard against empty LLM responses
            if not response:
                print(f"DEBUG: Entity extraction returned empty response, falling back to original query terms.")
                return [w for w in query.replace('?', '').split() if len(w) > 4][:5]

            # Try to extract a JSON array even from messy responses
            bracket_start = response.find('[')
            bracket_end = response.rfind(']')
            if bracket_start != -1 and bracket_end != -1:
                response = response[bracket_start:bracket_end + 1]

            entities = json.loads(response)
            if isinstance(entities, list):
                return entities[:5]
        except Exception as e:
            print(f"DEBUG: KBPedia entity extraction failed: {e}. Falling back to original query terms.")

        return [query]

    def find_concepts(self, search_term: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Fulltext search on KBPediaConcept nodes.
        Fuzzy fallback: if the exact phrase returns nothing, tries only the LONGEST
        word in the phrase (most specific/rare) rather than all words left-to-right.
        This prevents generic words like 'skeleton' from matching off-topic concepts
        like 'skeleton tarantula' when the original intent was 'animal skeleton'.
        """
        results = self._fulltext_search(search_term, limit)

        # Fuzzy fallback: use only the longest (most specific) word
        if not results and ' ' in search_term:
            words = [w for w in search_term.split() if len(w) > 3]
            if words:
                # Sort by length descending — longer word = more specific/rare
                longest_word = sorted(words, key=len, reverse=True)[0]
                print(f"DEBUG: KBPedia exact search failed for '{search_term}'. Trying longest-word fallback: '{longest_word}'", flush=True)
                results = self._fulltext_search(longest_word, limit=limit)

        return results

    def _fulltext_search(self, term: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Raw fulltext search against the index."""
        # Sanitize for Lucene
        safe_term = term.replace('"', '').replace('\\', '').replace('~', '').replace('*', '')
        safe_term = re.sub(r'[+\-!(){}[\]^"~*?:\\/]', ' ', safe_term).strip()
        if not safe_term:
            return []

        try:
            records, _, _ = self.driver.execute_query(
                """
                CALL db.index.fulltext.queryNodes('kbpediaConceptIndex', $term)
                YIELD node, score
                RETURN node.uri AS uri, node.name AS name, node.definition AS definition,
                       node.altLabels AS altLabels, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                term=safe_term,
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
            print(f"DEBUG: KBPedia concept search error for '{safe_term}': {e}")
            return []

    def get_neighborhood(self, uri: str) -> List[str]:
        """Get the ontology neighborhood of a concept (ancestors, children, siblings)."""
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (n:KBPediaConcept {uri: $uri})
                
                CALL (n) {
                    OPTIONAL MATCH (n)-[:SUBCLASS_OF*1..3]->(ancestor:KBPediaConcept)
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
                        triples.append(f"subclass of: {a['name']}")
                        if a.get("def"):
                            triples.append(f"ancestor definition: {a['def'][:150]}")
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
            return triples[:10]

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
            defn = (c.get("definition") or "")[:150]
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
            indices = json.loads(res)
            print(f"DEBUG: End-filter LLM kept indices: {indices} from {len(candidates)} candidates.", flush=True)
            return [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        except Exception as e:
            print(f"DEBUG: End-filter LLM call failed: {e}. Returning all candidates.", flush=True)
            return candidates

    def search(self, query_text: str, top_k: int = 5, original_query: str = "") -> Any:
        from .retriever import RetrieverResult, RetrieverResultItem

        if not self._verify_index():
            return RetrieverResult(items=[], metadata={})

        # `original_query` = full user prompt (with MCQ choices) for LLM filtering.
        # `query_text`     = extracted question stem, used for entity extraction.
        # Fall back to query_text if original_query was not provided.
        filter_query = original_query if original_query else query_text

        # ── Entity extraction ─────────────────────────────────────────────────
        if self.triple_filter:
            print(f"DEBUG: [KBPedia RIGOROUS mode] Querying for: '{query_text}'", flush=True)
            entities = self.extract_entities_from_query(query_text)
        else:
            print(f"DEBUG: [KBPedia FAST mode] Querying for: '{query_text}'", flush=True)
            entities = self._extract_entities_fast(query_text)
        print(f"DEBUG: Extracted entities for KBPedia: {entities}", flush=True)

        seen_uris = set()
        # In fast mode we buffer all candidates first, then do one LLM pass at the end.
        # In rigorous mode we filter per concept and build items directly.
        candidates = []  # for fast mode buffering: list of {name, uri, definition, triples}
        items = []       # final result items

        for concept_name in entities:
            if len(items) >= top_k and self.triple_filter:
                break

            matches = self.find_concepts(concept_name, limit=2)

            if not matches:
                print(f"DEBUG: KBPedia has no match for '{concept_name}', skipping.", flush=True)
                continue

            print(f"DEBUG: KBPedia matched {len(matches)} concept(s) for '{concept_name}': {[m['name'] for m in matches]}", flush=True)

            for match in matches:
                if len(items) >= top_k and self.triple_filter:
                    break

                uri = match["uri"]
                if uri in seen_uris:
                    continue
                seen_uris.add(uri)

                # Build raw triples
                raw_triples = []
                if match["definition"]:
                    raw_triples.append(f"definition: {match['definition'][:200]}")
                raw_triples.extend(self.get_neighborhood(uri))

                # Defensive filter: remove any triple that looks like an MCQ option
                mcq_pattern = re.compile(r'^[A-D][.):] ', re.IGNORECASE)
                raw_triples = [t for t in raw_triples if not mcq_pattern.match(t)]

                if not raw_triples:
                    print(f"DEBUG: No raw triples for '{match['name']}' ({uri}), skipping.", flush=True)
                    continue

                if self.triple_filter:
                    # RIGOROUS: LLM filters triples per concept
                    print(f"DEBUG: Built {len(raw_triples)} raw triple(s) for '{match['name']}'. Sending to LLM filter...", flush=True)
                    selected_triples = self.filter_triples_for_query(query_text, match["name"], raw_triples, original_query=filter_query)
                    print(f"DEBUG: LLM filter kept {len(selected_triples)}/{len(raw_triples)} triple(s) for '{match['name']}'", flush=True)

                    if not selected_triples:
                        continue

                    triples_text = "\n".join([f"  - {t}" for t in selected_triples])
                    content = f"KBPedia Concept: {match['name']}.\nRelevant Logical Facts:\n{triples_text}"
                    items.append(RetrieverResultItem(
                        content=content,
                        metadata={"source": "KBPedia", "entity": match["name"],
                                  "uri": uri, "score": 1.0, "triples": selected_triples},
                    ))
                else:
                    # FAST: buffer raw candidates for a single end-of-loop LLM pass
                    print(f"DEBUG: Buffered {len(raw_triples)} raw triple(s) for '{match['name']}'", flush=True)
                    candidates.append({
                        "name": match["name"],
                        "uri": uri,
                        "definition": match.get("definition", ""),
                        "triples": raw_triples,
                    })

        # ── Fast-mode: single LLM relevance filter over all candidates ─────────────
        if not self.triple_filter and candidates:
            filtered_candidates = self._filter_concepts_once(query_text, candidates, original_query=filter_query)
            print(f"DEBUG: End-filter kept {len(filtered_candidates)}/{len(candidates)} concepts.", flush=True)
            for c in filtered_candidates[:top_k]:
                triples_text = "\n".join([f"  - {t}" for t in c["triples"]])
                content = f"KBPedia Concept: {c['name']}.\nRelevant Logical Facts:\n{triples_text}"
                items.append(RetrieverResultItem(
                    content=content,
                    metadata={"source": "KBPedia", "entity": c["name"],
                              "uri": c["uri"], "score": 1.0, "triples": c["triples"]},
                ))

        print(f"DEBUG: KBPedia found {len(items)} grounded concept contexts.", flush=True)
        return RetrieverResult(items=items, metadata={})

