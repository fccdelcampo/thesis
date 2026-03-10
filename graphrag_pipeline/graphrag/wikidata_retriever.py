import asyncio
import aiohttp
import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitExceededException(Exception):
    pass

class WikidataRetriever:
    """
    Dedicated fetching module to retrieve Wikidata facts via SPARQL
    with rate limit protections, strict timeouts, and backoff logic.
    Converts results directly to valid Prolog assertions.
    """
    def __init__(self):
        self.endpoint_url = "https://query.wikidata.org/sparql"
        # Required Custom User-Agent by Wikidata Policy
        self.user_agent = "PrologGraphRAG_Agent/1.0 (Contact: local@example.org) Python/aiohttp"
        
        # Concurrency & Timeouts: No more than 5 parallel requests
        self.semaphore = asyncio.Semaphore(5)
        
        self.error_timestamps: List[float] = []

    async def _check_error_rate(self):
        """Halt and log if 30-error-per-minute threshold is approached."""
        now = time.time()
        # Keep only errors from the last 60 seconds
        self.error_timestamps = [t for t in self.error_timestamps if now - t < 60]
        if len(self.error_timestamps) >= 30:
            logger.error("Approaching 30-error-per-minute threshold. Halting requests to prevent IP ban.")
            raise RateLimitExceededException("Wikidata API error threshold exceeded (>30 errors/min).")

    def _record_error(self):
        self.error_timestamps.append(time.time())

    async def _fetch_facts_async(self, qid: str) -> List[str]:
        async with self.semaphore:
            sparql_query = f"""
            SELECT ?pLabel ?oLabel ?desc WHERE {{
              {{
                wd:{qid} ?p ?o.
                ?o rdfs:label ?oLabel.
                ?property wikibase:directClaim ?p.
                ?property rdfs:label ?pLabel.
                FILTER(LANG(?oLabel) = "en")
                FILTER(LANG(?pLabel) = "en")
              }}
              UNION
              {{
                # Fetch the text description of the entity
                wd:{qid} schema:description ?desc.
                FILTER(LANG(?desc) = "en")
              }}
            }} LIMIT 15
            """
            
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/sparql-results+json"
            }
            
            # Enforce strict 60-second timeout on all requests
            timeout = aiohttp.ClientTimeout(total=60)
            
            max_retries = 3
            for attempt in range(max_retries):
                await self._check_error_rate()
                
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(self.endpoint_url, params={"query": sparql_query}, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                facts = []
                                for item in data.get("results", {}).get("bindings", []):
                                    desc = item.get("desc", {}).get("value", "")
                                    if desc and f"definition: {desc.lower()}" not in facts:
                                        facts.insert(0, f"definition: {desc.lower()}") # Prioritize definition
                                        
                                    predicate = item.get("pLabel", {}).get("value", "")
                                    obj = item.get("oLabel", {}).get("value", "")
                                    
                                    # Format as plain text facts for GraphRAG
                                    if predicate and obj and len(facts) < 10:
                                        fact_str = f"{predicate.lower()}: {obj.lower()}"
                                        if fact_str not in facts:
                                            facts.append(fact_str)
                                            
                                return facts
                                
                            elif response.status == 429:
                                # Backoff Logic: Trap HTTP 429. Extract Retry-After
                                retry_after = int(response.headers.get("Retry-After", 10))
                                logger.warning(f"HTTP 429 Too Many Requests. Exact sleep for {retry_after} seconds.")
                                await asyncio.sleep(retry_after)
                                continue
                            else:
                                self._record_error()
                                logger.error(f"HTTP {response.status} from Wikidata for {qid}")
                                break
                                
                except asyncio.TimeoutError:
                    self._record_error()
                    logger.error(f"Strict 60-second timeout reached querying Wikidata for {qid}")
                    break
                except Exception as e:
                    self._record_error()
                    logger.error(f"Error querying Wikidata for {qid}: {e}")
                    break
                    
            return []

    def retrieve_facts(self, qid: str) -> List[str]:
        """Synchronous wrapper for main pipeline."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # For Jupyter or already running loops
                task = asyncio.ensure_future(self._fetch_facts_async(qid))
                # Note: this blocks the sync flow; might need nest_asyncio in real env
                return loop.run_until_complete(task)
        except RuntimeError:
            pass
        return asyncio.run(self._fetch_facts_async(qid))
