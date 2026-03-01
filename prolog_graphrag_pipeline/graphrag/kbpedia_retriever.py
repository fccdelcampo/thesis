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

    def __init__(self, driver, llm=None, top_k: int = 4):
        self.driver  = driver
        self.llm     = llm
        self.top_k   = top_k

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

    def extract_entities(self, query: str) -> List[str]:
        """
        Hybrid entity extraction:
        1. Uses Regex to accurately pull exact MCQ choices (fast and prevents LLM formatting crashes).
        2. Uses LLM to extract explicit entities and implied conceptual terms from the question stem.
        3. Combines and prioritizes the lists.
        """
        # --- 1. Regex MCQ Extraction ---
        mcq_choices = [c.strip() for c in re.findall(r'\b[A-D][.)]\s*(.+?)(?=\s+\b[A-D][.)]|$)', query)]
        # Clean up any trailing punctuation from choices
        mcq_choices = [re.sub(r'[?!.]+$', '', c).strip().lower() for c in mcq_choices if c.strip()]
        
        # --- 2. LLM Conceptual Extraction ---
        llm_entities = []
        if self.llm:
            # We provide the FULL query (stem + options) to the LLM so it can deduce context
            # and extract underlying concepts from the answer choices as well.
            prompt = f"""You are extracting search terms for a knowledge base lookup.

Given this query (which may include multiple-choice options), extract the following:
1. **Explicit entities** mentioned in the question stem (e.g., "plant cell", "animal skeleton")
2. **Implied functional concepts** the question is asking about (e.g., "photosynthesis")
3. **Core scientific terms** from the answer options. If an option is a long sentence like "Dogs are better than cats", extract only the core entities ("Dogs", "cats").

Query: "{query}"

Output a plain JSON list of strings. Include 4-8 terms maximum.
Example: ["plant cell", "structural support", "cell wall", "chloroplast"]"""
            
            try:
                response = self.llm.invoke(prompt)
                res_text = (response.content if hasattr(response, 'content') else str(response)).strip()
                
                # Robust JSON extraction: Find first '[' and last ']'
                bracket_start = res_text.find('[')
                bracket_end = res_text.rfind(']')
                
                if bracket_start != -1 and bracket_end != -1:
                    json_str = res_text[bracket_start:bracket_end + 1]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        llm_entities = [str(item).lower() for item in parsed]
                else:
                    print(f"DEBUG: Entity extraction LLM returned no JSON array. Raw: {repr(res_text[:100])}")
            except Exception as e:
                print(f"DEBUG: KBPedia entity extraction LLM failed: {e}")

        # --- 3. Fallback ---
        # If LLM fails or is missing, use basic word splitting for the stem
        if not llm_entities:
            stem = re.sub(r'\b[A-D][.)]\s?', ' ', query)
            stem = re.sub(r'[?!,;:()\[\]]', ' ', stem).replace('\n', ' ')
            words = [w.strip('.').lower() for w in stem.split() if len(w) >= 5]
            llm_entities = words[:5]

        # --- 4. Combine and deduplicate ---
        unique_entities = []
        seen = set()
        
        # Prioritize MCQ choices first, then LLM concepts
        for phrase in mcq_choices + llm_entities:
            if phrase and phrase not in seen:
                seen.add(phrase)
                unique_entities.append(phrase)
                
        # We don't want to sort by length anymore because LLM gives high-quality short phrases too.
        # MCQ choices naturally stay at the top.
        
        return unique_entities[:10]

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
                        # Only keep the ancestor NAME (e.g. "subclass of: Physical Quantity").
                        # Ancestor definition blurbs are long encyclopedia text that bloats
                        # the Prolog generator context without helping code generation.
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

    def filter_triples_batch(self, query: str, concepts_data: List[Dict[str, Any]], original_query: str = "") -> Dict[str, List[str]]:
        """
        Use LLM to keep only logically relevant triples for MULTIPLE concepts at once.
        concepts_data is a list of dicts: {"name": str, "triples": List[str]}
        Returns a dictionary mapping concept_name to a list of filtered triples.
        """
        if not self.llm or not concepts_data:
            return {c["name"]: c["triples"][:10] for c in concepts_data}

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

            filtered = json.loads(res)
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
        print(f"DEBUG: [KBPedia] Querying for: '{query_text}'", flush=True)
        entities = self.extract_entities(query_text)
        print(f"DEBUG: Extracted entities for KBPedia: {entities}", flush=True)

        seen_uris = set()
        candidates = []  # list of {name, uri, definition, triples}
        items = []       # final result items

        # ── Wikidata fallback setup ────────────────────────────────────────────
        try:
            from .config import ENABLE_WIKIDATA_FALLBACK
        except ImportError:
            ENABLE_WIKIDATA_FALLBACK = False

        wikidata_fetcher = None
        if ENABLE_WIKIDATA_FALLBACK:
            try:
                from .wikidata_retriever import WikidataRetriever
                wikidata_fetcher = WikidataRetriever()
            except ImportError:
                print("DEBUG: WikidataRetriever not available. Skipping Wikidata fallback.", flush=True)

        for concept_name in entities:
            matches = self.find_concepts(concept_name, limit=2)

            if not matches:
                print(f"DEBUG: KBPedia has no match for '{concept_name}'.", flush=True)

                # ── Wikidata Fallback ──────────────────────────────────────
                if wikidata_fetcher:
                    print(f"DEBUG: [Wikidata Fallback] Searching Wikidata for '{concept_name}'...", flush=True)
                    wd_candidates = wikidata_fetcher.search_entity(concept_name, limit=5)

                    if not wd_candidates:
                        print(f"DEBUG: [Wikidata Fallback] No Wikidata results for '{concept_name}'. Skipping.", flush=True)
                        continue

                    # Select the correct Q-ID + relevant property categories in ONE LLM call
                    selected = None
                    extra_props = []  # additional P-codes from LLM-selected categories

                    if len(wd_candidates) == 1:
                        selected = wd_candidates[0]
                        print(f"DEBUG: [Wikidata Fallback] Single match: {selected['qid']} ({selected['label']})", flush=True)

                    if self.llm:
                        # Build candidate list
                        options_text = "\n".join(
                            [f"  [{i}] {c['qid']} - {c['label']}: {c['description']}" for i, c in enumerate(wd_candidates)]
                        )

                        # Build extended property category descriptions
                        from .wikidata_retriever import WikidataRetriever as WDR
                        categories_text = "\n".join(
                            [f"  - {cat}: {', '.join(pids)}" for cat, pids in WDR.EXTENDED_PROPERTIES.items()]
                        )

                        select_prompt = f"""You are disambiguating a Wikidata entity AND selecting which relationship types to fetch.

User question: "{filter_query}"
Entity to find: "{concept_name}"

TASK 1 — Entity Selection:
{options_text}

TASK 2 — Property Categories:
The system always fetches: instance of, subclass of, part of, has part, has effect.
Below are OPTIONAL extended categories. Pick ONLY those relevant to the user's question:
  - causality: has cause, has effect
  - sequence: follows, followed by
  - differentiation: different from, said to be the same as
  - academic: studied in, main subject
  - composition: has part, facet of
  - function: has use, product/material produced

Reply with a JSON object. Example: {{"index": 0, "categories": ["causality", "composition"]}}
If no extended categories are needed, use an empty list: {{"index": 2, "categories": []}}
Output ONLY the JSON object, nothing else."""

                        try:
                            raw = self.llm.invoke(select_prompt)
                            res = (raw.content if hasattr(raw, 'content') else str(raw)).strip()

                            # Strip markdown fences
                            if "```json" in res:
                                res = res.split("```json")[1].split("```")[0].strip()
                            elif "```" in res:
                                res = res.split("```")[1].split("```")[0].strip()

                            # Find JSON object
                            brace_start = res.find("{")
                            brace_end = res.rfind("}")
                            if brace_start != -1 and brace_end != -1:
                                res = res[brace_start:brace_end + 1]

                            parsed = json.loads(res)
                            idx = parsed.get("index", 0)
                            categories = parsed.get("categories", [])

                            if isinstance(idx, int) and 0 <= idx < len(wd_candidates):
                                selected = wd_candidates[idx]
                                print(f"DEBUG: [Wikidata Fallback] LLM selected index {idx}: {selected['qid']} ({selected['label']})", flush=True)

                            # Resolve category names to P-codes
                            if isinstance(categories, list):
                                for cat in categories:
                                    if cat in WDR.EXTENDED_PROPERTIES:
                                        extra_props.extend(WDR.EXTENDED_PROPERTIES[cat])
                                if extra_props:
                                    print(f"DEBUG: [Wikidata Fallback] LLM selected extended categories: {categories} → {extra_props}", flush=True)

                        except Exception as e:
                            print(f"DEBUG: [Wikidata Fallback] LLM selection failed: {e}. Using first candidate.", flush=True)

                    if not selected:
                        selected = wd_candidates[0]
                        print(f"DEBUG: [Wikidata Fallback] Defaulting to first candidate: {selected['qid']} ({selected['label']})", flush=True)

                    # Fetch structural triples from Wikidata (core + LLM-selected extended)
                    print(f"DEBUG: [Wikidata Fallback] Fetching structural facts for {selected['qid']}...", flush=True)
                    wd_triples = wikidata_fetcher.retrieve_structural_facts(selected['qid'], extra_properties=extra_props if extra_props else None)

                    if wd_triples:
                        # Add definition from Wikidata description
                        all_triples = []
                        if selected.get('description'):
                            all_triples.append(f"definition: {selected['description']}")
                        all_triples.extend(wd_triples)

                        print(f"DEBUG: [Wikidata Fallback] Retrieved {len(all_triples)} structural triple(s) for '{selected['label']}'", flush=True)
                        for t in all_triples:
                            print(f"  > {t}", flush=True)

                        wd_uri = f"wikidata:{selected['qid']}"
                        if wd_uri not in seen_uris:
                            seen_uris.add(wd_uri)
                            candidates.append({
                                "name": selected['label'],
                                "uri": wd_uri,
                                "definition": selected.get('description', ''),
                                "triples": all_triples,
                            })
                    else:
                        print(f"DEBUG: [Wikidata Fallback] No structural facts returned for {selected['qid']}. Skipping.", flush=True)
                # ── End Wikidata Fallback ──────────────────────────────────
                continue

            print(f"DEBUG: KBPedia matched {len(matches)} concept(s) for '{concept_name}': {[m['name'] for m in matches]}", flush=True)

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

