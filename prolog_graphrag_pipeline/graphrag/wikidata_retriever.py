import requests
import time
from typing import List, Dict, Any, Optional

class WikidataRetriever:
    """
    Retrieves information from Wikidata (https://www.wikidata.org).
    Uses the Wikidata Search API to identify entities and then fetches claim data (triples).
    """
    
    SEARCH_API_URL = "https://www.wikidata.org/w/api.php"
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    def __init__(self, top_k: int = 3, language: str = "en"):
        self.top_k = top_k
        self.language = language
        self.headers = {
            'User-Agent': 'GraphRAG-Pipeline/1.0 (mailto:your_email@example.com)'
        }

    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Uses wbsearchentities to find Wikidata items matching the query.
        """
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': self.language,
            'search': query,
            'limit': self.top_k,
            'type': 'item'
        }
        
        try:
            response = requests.get(self.SEARCH_API_URL, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get('search', [])
        except Exception as e:
            print(f"Wikidata Search API Error: {e}")
            return []

    def get_entity_triples(self, entity_id: str) -> List[str]:
        """
        Fetches specific relationship triples for an entity using SPARQL.
        Focuses on: Instance Of (P31), Subclass Of (P279), Part Of (P361), 
        Replaced By (P1366), Replaces (P1365), and generic 'related to' for context.
        """
        sparql_query = f"""
        SELECT ?propLabel ?valLabel WHERE {{
          wd:{entity_id} ?p ?val .
          ?prop wikibase:directClaim ?p .
          ?val rdfs:label ?valLabel .
          ?prop rdfs:label ?propLabel .
          
          # Structural properties
          # P31: instance of, P279: subclass of, P361: part of,
          # P1366: replaced by, P1365: replaces, P527: has part
          #
          # Geographic properties
          # P17: country, P131: located in, P36: capital, P159: HQ location,
          # P1376: capital of, P625: coordinates, P2044: elevation
          #
          # Government/Organization
          # P6: head of government
          #
          # Education & Academic properties  
          # P69: educated at, P512: academic degree, P812: academic major,
          # P184: doctoral advisor, P802: student
          #
          # Biographical properties
          # P106: occupation, P108: employer, P19: place of birth,
          # P20: place of death, P27: country of citizenship,
          # P569: date of birth, P570: date of death
          #
          # Academic/Professional
          # P101: field of work, P800: notable work, P166: award received
          #
          # Temporal
          # P571: inception/founded date, P856: official website
          VALUES ?p {{
            wdt:P31 wdt:P279 wdt:P361 wdt:P1366 wdt:P1365 wdt:P527
            wdt:P17 wdt:P131 wdt:P36 wdt:P159 wdt:P1376 wdt:P2044
            wdt:P6
            wdt:P69 wdt:P512 wdt:P812 wdt:P184 wdt:P802
            wdt:P106 wdt:P108 wdt:P19 wdt:P20 wdt:P27 wdt:P569 wdt:P570
            wdt:P101 wdt:P800 wdt:P166
            wdt:P571 wdt:P856
          }}
          
          FILTER(LANG(?valLabel) = "{self.language}")
          FILTER(LANG(?propLabel) = "{self.language}")
        }} LIMIT 25
        """
        
        # If specific query returns nothing, we could fallback to generic, but let's stick to high-quality specific ones first.
        
        params = {
            'query': sparql_query,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.SPARQL_ENDPOINT, params=params, headers=self.headers)
            if response.status_code == 429:
                time.sleep(1) # Simple backoff
                response = requests.get(self.SPARQL_ENDPOINT, params=params, headers=self.headers)
                
            response.raise_for_status()
            data = response.json()
            
            bindings = data.get('results', {}).get('bindings', [])
            triples = []
            for b in bindings:
                prop = b.get('propLabel', {}).get('value', 'Unknown')
                val = b.get('valLabel', {}).get('value', 'Unknown')
                # Format: "Entity -[Property]-> Value"
                triples.append(f"{prop}: {val}")
                
            return triples
        except Exception as e:
            print(f"Wikidata SPARQL Error for {entity_id}: {e}")
            return []

    def search(self, query_text: str, top_k: int = 5) -> Any:
        # Duck-typing the RetrieverResult interface
        from .retriever import RetrieverResult, RetrieverResultItem
        
        print(f"DEBUG: Querying Wikidata for: '{query_text}'")
        
        # Handle "OR" queries from expansion
        if " OR " in query_text:
            parts = [p.strip().replace('"', '') for p in query_text.split(" OR ")]
        else:
            parts = [query_text.strip().replace('"', '')]
        
        # 1. Interleaved Search (Round-Robin)
        # Fetch results for ALL parts first
        results_map = {} # part_index -> [results]
        for i, part in enumerate(parts):
            if not part: continue
            res = self.search_entities(part)
            results_map[i] = res
            
        # Interleave
        all_entities = []
        seen_ids = set()
        
        for k in range(top_k): # Iterate enough times to fill top_k
            found_new = False
            for i in range(len(parts)):
                if i in results_map and len(results_map[i]) > 0:
                    ent = results_map[i].pop(0) # Pop the first result for this term
                    eid = ent.get('id')
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        # Reciprocal Rank Score based on overall position
                        # Base score 1.0, decay by position
                        ent['dynamic_score'] = 1.0 / (len(all_entities) + 1)
                        all_entities.append(ent)
                        found_new = True
                        if len(all_entities) >= top_k:
                            break
            if len(all_entities) >= top_k:
                break
            if not found_new:
                break # No more results from any term
        
        items = []
        for ent in all_entities:
            eid = ent.get('id')
            label = ent.get('label')
            desc = ent.get('description', 'No description')
            
            # 2. Get Triples for top matches
            triples = self.get_entity_triples(eid)
            
            # 3. Format as Context Item (Clean, no QID in text)
            # CRITICAL: Include triples in content so LLM sees them!
            triples_text = "\n".join([f"  - {t}" for t in triples])
            content = f"Wikidata Entity: {label}.\nDescription: {desc}.\nFacts:\n{triples_text}"
            
            metadata = {
                "source": "Wikidata",
                "entity": label,
                "id": eid,
                "score": ent.get('dynamic_score', 1.0),
                "triples": triples
            }
            
            items.append(RetrieverResultItem(content=content, metadata=metadata))
            
        print(f"DEBUG: Wikidata found {len(items)} items.")
        return RetrieverResult(items=items, metadata={})
