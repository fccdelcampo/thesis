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
    
    # Property Groups for Dynamic Retrieval
    # Comprehensive Educational Property Set
    # Covers Math, Science, Logic, Literature, History, and General Knowledge
    EDUCATIONAL_PROPERTIES = [
        # --- IDENTITY & DEFINITION ---
        "wdt:P31",    # instance of (e.g., human, chemical element)
        "wdt:P279",   # subclass of (e.g., physicist -> scientist)
        "wdt:P361",   # part of (e.g., brain -> central nervous system)
        "wdt:P1343",  # described by source (encyclopedias)
        "wdt:P18",    # image
        
        # --- MATH & LOGIC ---
        "wdt:P2562",  # formula (e.g., E=mc^2)
        "wdt:P461",   # opposite of (e.g., good <-> bad)
        "wdt:P112",   # founded by / derived from (e.g., theorem prover)
        "wdt:P2579",  # studied by (e.g., logic)
        "wdt:P155",   # follows
        "wdt:P156",   # followed by
        
        # --- SCIENCE & CAUSALITY ---
        "wdt:P828",   # has cause
        "wdt:P1542",  # has effect
        "wdt:P527",   # has part (constituents)
        "wdt:P61",    # discoverer or inventor
        "wdt:P366",   # has use
        "wdt:P1535",  # used by
        "wdt:P495",   # country of origin (inventions, styles)
        
        # --- HUMANITIES (LIT, ARTS, HISTORY) ---
        "wdt:P136",   # genre
        "wdt:P50",    # author
        "wdt:P577",   # publication date
        "wdt:P674",   # characters
        "wdt:P1344",  # participant in (events)
        
        # --- BIOGRAPHICAL & CONTEXT ---
        "wdt:P569", "wdt:P570",  # birth/death dates
        "wdt:P19", "wdt:P20",    # birth/death places
        "wdt:P69",               # educated at
        "wdt:P106",              # occupation
        "wdt:P101",              # field of work
        "wdt:P108",              # employer
        "wdt:P800",              # notable work
        "wdt:P166",              # award received
        "wdt:P184",              # doctoral advisor
        
        # --- ORGANIZATION & GEOGRAPHY (CONTEXT) ---
        "wdt:P571",   # inception
        "wdt:P159",   # HQ location
        "wdt:P131",   # located in
        "wdt:P17",    # country (context)
        "wdt:P36",    # capital
    ]

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
        Fetches relevant educational relationship triples for an entity using a comprehensive property set.
        """
        
        # Join Unified Property Set for SPARQL
        values_clause = " ".join(self.EDUCATIONAL_PROPERTIES)

        sparql_query = f"""
        SELECT ?propLabel ?valLabel WHERE {{
          wd:{entity_id} ?p ?val .
          ?prop wikibase:directClaim ?p .
          ?val rdfs:label ?valLabel .
          ?prop rdfs:label ?propLabel .
          
          VALUES ?p {{ {values_clause} }}
          
          FILTER(LANG(?valLabel) = "{self.language}")
          FILTER(LANG(?propLabel) = "{self.language}")
        }} LIMIT 40
        """
        
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
        
        # Handle "OR" queries from expansion if any
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
            
            # 2. Get Triples using Unified List
            triples = self.get_entity_triples(eid)
            
            # 3. Format as Context Item
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
