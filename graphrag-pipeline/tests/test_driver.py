"""
Unit and integration tests for graphrag/driver.py

Tests model initialization and KG pipeline setup.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag.graphrag_driver import (
    initialize_models,
    setup_kg_pipeline,
    generate_answer
)


# ===== UNIT TESTS =====

class TestInitializeModels:
    """Test the initialize_models function."""
    
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.OllamaEmbeddings')
    def test_initialize_models_returns_tuple(self, mock_embeddings, mock_llm):
        """initialize_models should return a tuple of two objects."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        result = initialize_models()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.OllamaEmbeddings')
    def test_initialize_models_creates_llm(self, mock_embeddings, mock_llm):
        """initialize_models should create an OllamaLLM instance."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        llm, _ = initialize_models()
        
        mock_llm.assert_called_once()
        assert llm == mock_llm_instance
    
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.OllamaEmbeddings')
    def test_initialize_models_creates_embedder(self, mock_embeddings, mock_llm):
        """initialize_models should create an OllamaEmbeddings instance."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        _, embedder = initialize_models()
        
        mock_embeddings.assert_called_once()
        assert embedder == mock_embeddings_instance
    
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.OllamaEmbeddings')
    def test_initialize_models_llm_uses_llama3_model(self, mock_embeddings, mock_llm):
        """initialize_models should create LLM with llama3 model."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        initialize_models()
        
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs['model_name'] == 'llama3'
    
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.OllamaEmbeddings')
    def test_initialize_models_embedder_uses_llama3_model(self, mock_embeddings, mock_llm):
        """initialize_models should create embedder with llama3 model."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        initialize_models()
        
        call_kwargs = mock_embeddings.call_args[1]
        assert call_kwargs['model'] == 'llama3'


class TestSetupKgPipeline:
    """Test the setup_kg_pipeline function."""
    
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_returns_tuple(self, mock_pipeline):
        """setup_kg_pipeline should return a tuple of two pipelines."""
        mock_pipeline.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        result = setup_kg_pipeline(mock_llm, mock_embedder)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_creates_two_pipelines(self, mock_pipeline):
        """setup_kg_pipeline should create two SimpleKGPipeline instances."""
        mock_pipeline.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        setup_kg_pipeline(mock_llm, mock_embedder)
        
        assert mock_pipeline.call_count == 2
    
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_pdf_uses_from_pdf_true(self, mock_pipeline):
        """First pipeline should have from_pdf=True."""
        mock_pipeline.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        setup_kg_pipeline(mock_llm, mock_embedder)
        
        first_call_kwargs = mock_pipeline.call_args_list[0][1]
        assert first_call_kwargs['from_pdf'] is True
    
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_text_uses_from_pdf_false(self, mock_pipeline):
        """Second pipeline should have from_pdf=False."""
        mock_pipeline.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        setup_kg_pipeline(mock_llm, mock_embedder)
        
        second_call_kwargs = mock_pipeline.call_args_list[1][1]
        assert second_call_kwargs['from_pdf'] is False
    
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_uses_correct_schema(self, mock_pipeline):
        """Both pipelines should use 'EXTRACTED' schema."""
        mock_pipeline.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        setup_kg_pipeline(mock_llm, mock_embedder)
        
        for call in mock_pipeline.call_args_list:
            assert call[1]['schema'] == 'EXTRACTED'
    
    @patch('graphrag.driver.FixedSizeSplitter')
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_setup_kg_pipeline_creates_text_splitter(self, mock_pipeline, mock_splitter):
        """setup_kg_pipeline should create a FixedSizeSplitter."""
        mock_pipeline.return_value = MagicMock()
        mock_splitter.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_embedder = MagicMock()
        
        setup_kg_pipeline(mock_llm, mock_embedder)
        
        mock_splitter.assert_called_once()


class TestGenerateAnswer:
    """Test the generate_answer function."""
    
    @patch('graphrag.driver.generate')
    def test_generate_answer_calls_generate_function(self, mock_generate):
        """generate_answer should call the generate function."""
        mock_generate.return_value = {"answer": "test", "context": "context"}
        mock_query = "test query"
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        
        generate_answer(mock_query, mock_retriever, mock_llm)
        
        mock_generate.assert_called_once_with(mock_llm, mock_retriever, mock_query)
    
    @patch('graphrag.driver.generate')
    def test_generate_answer_returns_dict(self, mock_generate):
        """generate_answer should return the result from generate function."""
        expected_result = {"answer": "test answer", "context": "test context"}
        mock_generate.return_value = expected_result
        
        result = generate_answer("query", MagicMock(), MagicMock())
        
        assert result == expected_result
    
    @patch('graphrag.driver.generate')
    def test_generate_answer_passes_correct_order(self, mock_generate):
        """generate_answer should pass llm and retriever in correct order."""
        mock_generate.return_value = {"answer": "test", "context": "context"}
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_query = "test query"
        
        generate_answer(mock_query, mock_retriever, mock_llm)
        
        call_args = mock_generate.call_args[0]
        assert call_args[0] == mock_llm
        assert call_args[1] == mock_retriever
        assert call_args[2] == mock_query


# ===== INTEGRATION TESTS =====

class TestInitializeAndSetupIntegration:
    """Integration tests combining model initialization and pipeline setup."""
    
    @patch('graphrag.driver.OllamaEmbeddings')
    @patch('graphrag.driver.OllamaLLM')
    @patch('graphrag.driver.SimpleKGPipeline')
    def test_initialize_then_setup_pipeline(self, mock_pipeline, mock_llm, mock_embeddings):
        """Test initialize_models followed by setup_kg_pipeline."""
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_pipeline.return_value = MagicMock()
        
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        llm, embedder = initialize_models()
        kg_pdf, kg_text = setup_kg_pipeline(llm, embedder)
        
        assert llm == mock_llm_instance
        assert embedder == mock_embeddings_instance
        assert mock_pipeline.call_count == 2


class TestGenerateAnswerIntegration:
    """Integration tests for generate_answer."""
    
    @patch('graphrag.driver.generate')
    def test_generate_answer_with_multiple_calls(self, mock_generate):
        """Test generate_answer can be called multiple times."""
        mock_generate.return_value = {"answer": "answer", "context": "context"}
        
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        
        result1 = generate_answer("query1", mock_retriever, mock_llm)
        result2 = generate_answer("query2", mock_retriever, mock_llm)
        
        assert result1 == result2
        assert mock_generate.call_count == 2
