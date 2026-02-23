"""
Grounded KBPedia Retriever — Neo4j Edition
Queries KBPedia concepts stored in Neo4j (:KBPediaConcept nodes)
and uses an LLM to filter only query-relevant facts for the Prolog pipeline.
"""

import json
from typing import List, Dict, Any, Optional


class KBPediaRetriever:
    """
    Retrieves grounded knowledge from KBPedia concepts stored in Neo4j.
    Uses fulltext search to find concepts, traverses the ontology hierarchy,
    and applies LLM-based filtering to return only logically relevant facts.
    """

    _index_verified = False

    def __init__(self, driver, llm=None, top_k: int = 3):
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
        """Use LLM to extract key entities from the user query."""
        if not self.llm:
            return [w for w in query.replace('?', '').split() if len(w) > 4][:3]

        prompt = f"""Extract the core, specific entities from this query for a knowledge base lookup.
Query: "{query}"
Output a plain JSON list of strings representing the entities. Do not include anything else.
Example: ["cell wall", "animal skeleton"]"""
        try:
            response = self.llm.invoke(prompt).content.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            entities = json.loads(response)
            if isinstance(entities, list):
                return entities
        except Exception as e:
            print(f"DEBUG: KBPedia entity extraction failed: {e}")

        return [query]

    def find_concepts(self, search_term: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Fulltext search on KBPediaConcept nodes."""
        # Sanitize search term for Lucene
        safe_term = search_term.replace('"', '').replace('\\', '').replace('~', '').replace('*', '')
        if not safe_term.strip():
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
            print(f"DEBUG: KBPedia concept search error: {e}")
            return []

    def get_neighborhood(self, uri: str) -> List[str]:
        """Get the ontology neighborhood of a concept: parents, siblings, children."""
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (n:KBPediaConcept {uri: $uri})
                OPTIONAL MATCH (n)-[:SUBCLASS_OF]->(parent:KBPediaConcept)
                OPTIONAL MATCH (child:KBPediaConcept)-[:SUBCLASS_OF]->(n)
                OPTIONAL MATCH (sibling:KBPediaConcept)-[:SUBCLASS_OF]->(parent)
                WHERE sibling.uri <> n.uri
                WITH n,
                     collect(DISTINCT {role: 'parent', name: parent.name, def: parent.definition}) AS parents,
                     collect(DISTINCT {role: 'child', name: child.name, def: child.definition}) AS children,
                     collect(DISTINCT {role: 'sibling', name: sibling.name, def: sibling.definition}) AS siblings
                RETURN parents, children, siblings[0..5] AS siblings
                """,
                uri=uri,
                database_="neo4j",
            )

            triples = []
            if records:
                row = records[0]
                for p in row["parents"]:
                    if p["name"]:
                        triples.append(f"subclass of: {p['name']}")
                        if p["def"]:
                            triples.append(f"parent definition: {p['def'][:150]}")
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
Example: ["subclass of: Structural Component", "parent definition: provides support"]
If none are relevant, output an empty list [].
"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()

            filtered = json.loads(res)
            if isinstance(filtered, list):
                return filtered
        except Exception as e:
            print(f"DEBUG: Triple filtering failed: {e}")

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
