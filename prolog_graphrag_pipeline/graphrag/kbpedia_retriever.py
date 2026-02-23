"""
Grounded KBPedia Retriever — Neo4j Edition
Queries KBPedia concepts stored in Neo4j (:KBPediaConcept nodes)
and uses an LLM to filter only query-relevant facts for the Prolog pipeline.
"""

import json
import re
from typing import List, Dict, Any, Optional


class KBPediaRetriever:
    """
    Retrieves grounded knowledge from KBPedia concepts stored in Neo4j.
    Uses fulltext search to find concepts, traverses the ontology hierarchy,
    and applies LLM-based filtering to return only logically relevant facts.
    """

    _index_verified = False

    def __init__(self, driver, llm=None, top_k: int = 8):
        self.driver = driver
        self.llm = llm
        self.top_k = top_k

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

    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Use LLM to extract entities from the user query.
        Extracts explicit entities, functional/conceptual terms, AND MCQ choices.
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
                return entities[:8]
        except Exception as e:
            print(f"DEBUG: KBPedia entity extraction failed: {e}. Falling back to original query terms.")

        return [query]

    def find_concepts(self, search_term: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Fulltext search on KBPediaConcept nodes with fuzzy fallback.
        If exact search returns nothing, tries individual words.
        """
        results = self._fulltext_search(search_term, limit)

        # Fuzzy fallback: if no results, try individual words
        if not results and ' ' in search_term:
            words = [w for w in search_term.split() if len(w) > 3]
            for word in words:
                word_results = self._fulltext_search(word, limit=2)
                if word_results:
                    results.extend(word_results)
                    if len(results) >= limit:
                        break
            # Deduplicate by URI
            seen = set()
            deduped = []
            for r in results:
                if r["uri"] not in seen:
                    seen.add(r["uri"])
                    deduped.append(r)
            results = deduped[:limit]

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
        """Get the ontology neighborhood of a concept: parents, siblings, children (up to 3 hops)."""
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (n:KBPediaConcept {uri: $uri})
                OPTIONAL MATCH (n)-[:SUBCLASS_OF*1..2]->(ancestor:KBPediaConcept)
                OPTIONAL MATCH (child:KBPediaConcept)-[:SUBCLASS_OF]->(n)
                OPTIONAL MATCH (n)-[:SUBCLASS_OF]->(parent:KBPediaConcept)<-[:SUBCLASS_OF]-(sibling:KBPediaConcept)
                WHERE sibling.uri <> n.uri
                WITH n,
                     collect(DISTINCT {role: 'ancestor', name: ancestor.name, def: ancestor.definition}) AS ancestors,
                     collect(DISTINCT {role: 'child', name: child.name, def: child.definition}) AS children,
                     collect(DISTINCT {role: 'sibling', name: sibling.name, def: sibling.definition}) AS siblings
                RETURN ancestors[0..5] AS ancestors, children[0..8] AS children, siblings[0..5] AS siblings
                """,
                uri=uri,
                database_="neo4j",
            )

            triples = []
            if records:
                row = records[0]
                for a in row["ancestors"]:
                    if a["name"]:
                        triples.append(f"subclass of: {a['name']}")
                        if a["def"]:
                            triples.append(f"ancestor definition: {a['def'][:150]}")
                for c in row["children"]:
                    if c["name"]:
                        triples.append(f"has subclass: {c['name']}")
                for s in row["siblings"]:
                    if s["name"]:
                        triples.append(f"sibling concept: {s['name']}")
            return triples
        except Exception as e:
            print(f"DEBUG: KBPedia neighborhood error: {e}")
            return []

    def filter_triples_for_query(self, query: str, entity_label: str, triples: List[str]) -> List[str]:
        """Use LLM to keep only logically relevant triples."""
        if not self.llm or not triples:
            return triples[:10]

        triples_text = "\n".join([f"- {t}" for t in triples])
        prompt = f"""Task: You are deeply integrated into a Prolog GraphRAG pipeline.
Below is a list of known facts (triples) about the entity '{entity_label}'.
Your job is to strictly filter these facts and select ONLY the ones that are logically relevant or useful to answer the user's query.
Irrelevant facts will poison the logic engine, so drop anything that doesn't help.

User Query: "{query}"

Facts about {entity_label}:
{triples_text}

Output format: A JSON list of strings containing exactly the original facts you deemed relevant.
Example: ["subclass of: Structural Component", "ancestor definition: provides support"]
If none are relevant, output an empty list [].
"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()

            # Guard against empty LLM responses
            if not res:
                print(f"DEBUG: Triple filtering returned empty response, using raw triples.")
                return triples[:5]

            # Try to extract a JSON array even from messy responses
            bracket_start = res.find('[')
            bracket_end = res.rfind(']')
            if bracket_start != -1 and bracket_end != -1:
                res = res[bracket_start:bracket_end + 1]

            filtered = json.loads(res)
            if isinstance(filtered, list):
                return filtered
        except Exception as e:
            print(f"DEBUG: Triple filtering failed: {e}. Using raw triples.")

        return triples[:5]

    def search(self, query_text: str, top_k: int = 5) -> Any:
        from .retriever import RetrieverResult, RetrieverResultItem

        if not self._verify_index():
            return RetrieverResult(items=[], metadata={})

        print(f"DEBUG: Querying KBPedia (Neo4j) for: '{query_text}'")

        entities = self.extract_entities_from_query(query_text)
        print(f"DEBUG: Extracted entities for KBPedia: {entities}")

        items = []
        seen_uris = set()

        for concept_name in entities:
            if len(items) >= top_k:
                break

            matches = self.find_concepts(concept_name, limit=2)

            if not matches:
                print(f"DEBUG: KBPedia has no match for '{concept_name}', skipping.")
                continue

            for match in matches:
                if len(items) >= top_k:
                    break

                uri = match["uri"]
                if uri in seen_uris:
                    continue
                seen_uris.add(uri)

                # Build fact list from definition + neighborhood
                raw_triples = []
                if match["definition"]:
                    raw_triples.append(f"definition: {match['definition'][:200]}")
                raw_triples.extend(self.get_neighborhood(uri))

                if not raw_triples:
                    continue

                filtered_triples = self.filter_triples_for_query(query_text, match["name"], raw_triples)

                if not filtered_triples:
                    continue

                triples_text = "\n".join([f"  - {t}" for t in filtered_triples])
                content = f"KBPedia Concept: {match['name']}.\nRelevant Logical Facts:\n{triples_text}"

                metadata = {
                    "source": "KBPedia",
                    "entity": match["name"],
                    "uri": uri,
                    "score": 1.0,
                    "triples": filtered_triples,
                }

                items.append(RetrieverResultItem(content=content, metadata=metadata))

        print(f"DEBUG: KBPedia found {len(items)} grounded concept contexts.")
        return RetrieverResult(items=items, metadata={})
