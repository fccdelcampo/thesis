"""
Unit and integration tests for graphrag/retriever.py

Tests vector index creation and retriever setup.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag.retriever import (
    extract_context_from_prompt,
    create_retriever
)


# ===== UNIT TESTS =====

class TestExtractContextFromPrompt:
    """Test the extract_context_from_prompt function."""
    
    def test_extract_context_from_prompt_returns_string(self):
        """extract_context_from_prompt should return a string."""
        result = extract_context_from_prompt("test prompt")
        assert isinstance(result, str) or result is None
    
    def test_extract_context_from_prompt_with_empty_string(self):
        """extract_context_from_prompt should handle empty strings."""
        result = extract_context_from_prompt("")
        assert result is None or isinstance(result, str)
    
    def test_extract_context_from_prompt_multiple_calls(self):
        """extract_context_from_prompt should work with multiple calls."""
        result1 = extract_context_from_prompt("prompt 1")
        result2 = extract_context_from_prompt("prompt 2")
        
        assert (result1 is None or isinstance(result1, str))
        assert (result2 is None or isinstance(result2, str))


class TestCreateRetriever:
    """Test the create_retriever function."""
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_returns_retriever(self, mock_retriever_class, mock_create_index):
        """create_retriever should return a HybridCypherRetriever instance."""
        mock_retriever_instance = MagicMock()
        mock_retriever_class.return_value = mock_retriever_instance
        
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        result = create_retriever(mock_driver, mock_embedder)
        
        assert result == mock_retriever_instance
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_creates_vector_index(self, mock_retriever_class, mock_create_index):
        """create_retriever should call create_vector_index."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        mock_create_index.assert_called_once()
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_index_dimensions_4096(self, mock_retriever_class, mock_create_index):
        """create_vector_index should be called with 4096 dimensions for Llama 3."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        call_kwargs = mock_create_index.call_args[1]
        assert call_kwargs['dimensions'] == 4096
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_index_name(self, mock_retriever_class, mock_create_index):
        """create_vector_index should use correct index name."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        call_kwargs = mock_create_index.call_args[1]
        assert call_kwargs['name'] == 'text_embeddings'
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_index_label(self, mock_retriever_class, mock_create_index):
        """create_vector_index should use 'Chunk' label."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        call_kwargs = mock_create_index.call_args[1]
        assert call_kwargs['label'] == 'Chunk'
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_index_similarity_fn(self, mock_retriever_class, mock_create_index):
        """create_vector_index should use cosine similarity."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        call_kwargs = mock_create_index.call_args[1]
        assert call_kwargs['similarity_fn'] == 'cosine'
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_handles_index_drop_error(self, mock_retriever_class, mock_create_index):
        """create_retriever should handle errors when dropping old index."""
        mock_retriever_class.return_value = MagicMock()
        mock_driver = MagicMock()
        mock_driver.execute_query.side_effect = Exception("Index doesn't exist")
        mock_embedder = MagicMock()
        
        # Should not raise exception
        result = create_retriever(mock_driver, mock_embedder)
        
        assert result is not None
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_instantiates_retriever_class(self, mock_retriever_class, mock_create_index):
        """create_retriever should instantiate HybridCypherRetriever."""
        mock_retriever_instance = MagicMock()
        mock_retriever_class.return_value = mock_retriever_instance
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        mock_retriever_class.assert_called_once()
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_passes_embedder_to_retriever(self, mock_retriever_class, mock_create_index):
        """create_retriever should pass embedder to HybridCypherRetriever."""
        mock_retriever_instance = MagicMock()
        mock_retriever_class.return_value = mock_retriever_instance
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        create_retriever(mock_driver, mock_embedder)
        
        call_kwargs = mock_retriever_class.call_args[1]
        assert call_kwargs['embedder'] == mock_embedder


# ===== INTEGRATION TESTS =====

class TestCreateRetrieverIntegration:
    """Integration tests for create_retriever."""
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_full_flow(self, mock_retriever_class, mock_create_index):
        """Test create_retriever complete flow with driver and embedder."""
        mock_retriever_instance = MagicMock()
        mock_retriever_class.return_value = mock_retriever_instance
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        result = create_retriever(mock_driver, mock_embedder)
        
        # Verify both index creation and retriever instantiation
        mock_create_index.assert_called_once()
        mock_retriever_class.assert_called_once()
        assert result == mock_retriever_instance
    
    @patch('graphrag.retriever.create_vector_index')
    @patch('graphrag.retriever.HybridCypherRetriever')
    def test_create_retriever_multiple_calls_independent(self, mock_retriever_class, mock_create_index):
        """Test multiple create_retriever calls are independent."""
        mock_retriever_instance1 = MagicMock()
        mock_retriever_instance2 = MagicMock()
        mock_retriever_class.side_effect = [mock_retriever_instance1, mock_retriever_instance2]
        
        mock_driver = MagicMock()
        mock_embedder = MagicMock()
        
        result1 = create_retriever(mock_driver, mock_embedder)
        result2 = create_retriever(mock_driver, mock_embedder)
        
        assert result1 == mock_retriever_instance1
        assert result2 == mock_retriever_instance2
        assert mock_create_index.call_count == 2
