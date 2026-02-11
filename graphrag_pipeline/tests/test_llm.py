"""
Unit and integration tests for llm.py

Tests the generate function for LLM-based answer generation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm import generate


class TestGenerate:
    """Test the generate function."""
    
    def test_generate_returns_dict(self):
        """generate() should return a dictionary."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        with patch('llm.GraphRAG') as mock_graph_rag_class:
            mock_graph_rag_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.retriever_result = "test context"
            mock_result.answer = "test answer"
            
            mock_graph_rag_instance.search.return_value = mock_result
            mock_graph_rag_class.return_value = mock_graph_rag_instance
            
            result = generate(mock_llm, mock_retriever, "test query")
            
            assert isinstance(result, dict)
    
    def test_generate_dict_has_required_keys(self):
        """generate() should return dict with 'answer' and 'context' keys."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        with patch('llm.GraphRAG') as mock_graph_rag_class:
            mock_graph_rag_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.retriever_result = "retrieved context"
            mock_result.answer = "test answer"
            
            mock_graph_rag_instance.search.return_value = mock_result
            mock_graph_rag_class.return_value = mock_graph_rag_instance
            
            result = generate(mock_llm, mock_retriever, "test query")
            
            assert "answer" in result
            assert "context" in result
    
    def test_generate_with_different_queries(self):
        """generate() should work with different query strings."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        with patch('llm.GraphRAG') as mock_graph_rag_class:
            mock_graph_rag_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.retriever_result = "context"
            mock_result.answer = "answer"
            
            mock_graph_rag_instance.search.return_value = mock_result
            mock_graph_rag_class.return_value = mock_graph_rag_instance
            
            queries = ["What is AI?", "Tell me about ML", "Explain deep learning"]
            
            for query in queries:
                result = generate(mock_llm, mock_retriever, query)
                assert isinstance(result, dict)
                assert "answer" in result
                assert "context" in result
    
    def test_generate_calls_graph_rag_search(self):
        """generate() should call GraphRAG.search with correct parameters."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        with patch('llm.GraphRAG') as mock_graph_rag_class:
            mock_graph_rag_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.retriever_result = "context"
            mock_result.answer = "answer"
            
            mock_graph_rag_instance.search.return_value = mock_result
            mock_graph_rag_class.return_value = mock_graph_rag_instance
            
            query = "test query"
            generate(mock_llm, mock_retriever, query)
            
            mock_graph_rag_instance.search.assert_called_once()
            call_args = mock_graph_rag_instance.search.call_args
            assert query in call_args[0]


class TestGenerateIntegration:
    """Integration tests for generate() with mocked dependencies."""
    
    def test_generate_integration_with_rag_template(self):
        """Test generate() creates RagTemplate correctly."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        with patch('llm.RagTemplate') as mock_template:
            with patch('llm.GraphRAG') as mock_graph_rag_class:
                mock_graph_rag_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.retriever_result = "context"
                mock_result.answer = "answer"
                
                mock_graph_rag_instance.search.return_value = mock_result
                mock_graph_rag_class.return_value = mock_graph_rag_instance
                
                generate(mock_llm, mock_retriever, "test query")
                
                mock_template.assert_called_once()
    
    def test_generate_integration_returns_answer_and_context(self):
        """Integration test: verify returned answer and context from GraphRAG."""
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        expected_context = "This is the retrieved context."
        expected_answer = "This is the generated answer."
        
        with patch('llm.GraphRAG') as mock_graph_rag_class:
            mock_graph_rag_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.retriever_result = expected_context
            mock_result.answer = expected_answer
            
            mock_graph_rag_instance.search.return_value = mock_result
            mock_graph_rag_class.return_value = mock_graph_rag_instance
            
            result = generate(mock_llm, mock_retriever, "test query")
            
            assert result["answer"] == expected_answer
            assert result["context"] == expected_context
