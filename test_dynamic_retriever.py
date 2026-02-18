import os
import sys
import unittest.mock
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from graphrag_pipeline.graphrag.retriever import EnhancedRetriever
from prolog_graphrag_pipeline.graphrag.wikidata_retriever import WikidataRetriever
from neo4j_graphrag.types import RetrieverResult

# Mock LLM that returns JSON with NO category
class MockLLM:
    def __init__(self, responses):
        self.responses = responses
        
    def invoke(self, prompt):
        # Return the next response
        return MagicMock(content=self.responses.pop(0))

def test_educational_retrieval():
    print("=== Testing Unified Educational Wikidata Retrieval ===")
    
    # Start Patches
    patcher_init = unittest.mock.patch('neo4j_graphrag.retrievers.VectorCypherRetriever.__init__', return_value=None)
    patcher_search = unittest.mock.patch('neo4j_graphrag.retrievers.VectorCypherRetriever.search')
    
    mock_init = patcher_init.start()
    mock_search = patcher_search.start()
    
    # Configure mock local search to return empty result
    mock_search.return_value = RetrieverResult(items=[], metadata={})

    try:
        # 1. Setup Mocks
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        # Mock LLM responses for 3 queries (No categories!)
        llm_responses = [
            # Einstein
            """{"queries": ["Albert Einstein"]}""",
            # Relativity
            """{"queries": ["Theory of Relativity"]}""",
             # Logic (Concept)
            """{"queries": ["Logic"]}"""
        ]
        
        mock_llm = MockLLM(llm_responses)
        
        # Initialize Retriever
        retriever = EnhancedRetriever(
            driver=mock_driver,
            index_name="test_index",
            retrieval_query="",
            embedder=mock_embedder,
            llm=mock_llm
        )
        # Manually set attributes since __init__ is patched
        retriever.driver = mock_driver
        retriever.llm = mock_llm
        
        # ---------------------------------------------------------
        # Test 1: Biography / Person (Einstein)
        # ---------------------------------------------------------
        print("\n--- Query 1: 'Einstein' (Expect: Education/Work) ---")
        result1 = retriever.search("Einstein")
        
        found_bio = False
        for item in result1.items:
            if item.metadata.get("source") == "Wikidata":
                content = item.content
                # Check for Education props (P69) or Work (P106)
                if "educated at:" in content or "occupation:" in content or "doctoral advisor:" in content:
                    found_bio = True
                    print(f" Found Education/Bio properties for {item.metadata['entity']}")
                    
        if not found_bio:
            print("Did NOT find Bio/Education properties.")

        # ---------------------------------------------------------
        # Test 2: Science / Theory (Relativity)
        # ---------------------------------------------------------
        print("\n--- Query 2: 'Theory of Relativity' (Expect: Discovery/Formula/Parts) ---")
        result2 = retriever.search("Relativity")
        
        found_sci = False
        for item in result2.items:
            if item.metadata.get("source") == "Wikidata":
                content = item.content
                # Check for Discoverer (P61), Parts (P527), or Subclass (P279)
                if "discoverer or inventor:" in content or "subclass of:" in content or "has part:" in content:
                    found_sci = True
                    print(f"Found Science properties for {item.metadata['entity']}")
                else:
                    print(f"   (Content snippet: {content[:100]}...)")
                    
        if not found_sci:
            print("Did NOT find Science properties.")
            
        # ---------------------------------------------------------
        # Test 3: Abstract Concept (Logic)
        # ---------------------------------------------------------
        print("\n--- Query 3: 'Logic' (Expect: Definition/Studied By) ---")
        result3 = retriever.search("Logic")
        
        found_logic = False
        for item in result3.items:
            if item.metadata.get("source") == "Wikidata":
                content = item.content
                # Check for Studied by (P2579), Subclass (P279), or Description
                if "studied by:" in content or "subclass of:" in content:
                    found_logic = True
                    print(f"Found Logic/Abstract properties for {item.metadata['entity']}")
                    
        if not found_logic:
            print("Did NOT find Logic properties.")
            
    finally:
        patcher_init.stop()
        patcher_search.stop()

if __name__ == "__main__":
    test_educational_retrieval()
