"""
Unit and integration tests for graphrag/encoder.py

Tests functions for query encoding, context extraction, and document processing.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag.encoder import (
    generate_with_ollama,
    extract_query_and_context,
    process_pdf_documents,
    process_text_context,
    process_context
)


# ===== UNIT TESTS =====

class TestGenerateWithOllama:
    """Test the generate_with_ollama function."""
    
    @patch('graphrag.encoder.ollama.chat')
    def test_generate_with_ollama_success(self, mock_chat):
        """generate_with_ollama should return a string on successful response."""
        mock_response = {
            'message': {'content': '{"cleaned_query": "test", "extracted_context": "context"}'}
        }
        mock_chat.return_value = mock_response
        
        result = generate_with_ollama("test prompt")
        
        assert result is not None
        assert isinstance(result, str)
    
    @patch('graphrag.encoder.ollama.chat')
    def test_generate_with_ollama_exception_returns_none(self, mock_chat):
        """generate_with_ollama should return None when Ollama raises an exception."""
        mock_chat.side_effect = Exception("Connection error")
        
        result = generate_with_ollama("test prompt")
        
        assert result is None
    
    @patch('graphrag.encoder.ollama.chat')
    def test_generate_with_ollama_uses_correct_model(self, mock_chat):
        """generate_with_ollama should use the correct Ollama model."""
        mock_response = {'message': {'content': '{"test": "data"}'}}
        mock_chat.return_value = mock_response
        
        generate_with_ollama("test prompt")
        
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['model'] == 'llama3.2:3b-instruct-q8_0'
    
    @patch('graphrag.encoder.ollama.chat')
    def test_generate_with_ollama_json_format_requested(self, mock_chat):
        """generate_with_ollama should request JSON format."""
        mock_response = {'message': {'content': '{"test": "data"}'}}
        mock_chat.return_value = mock_response
        
        generate_with_ollama("test prompt")
        
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['format'] == 'json'
    
    @patch('graphrag.encoder.ollama.chat')
    def test_generate_with_ollama_zero_temperature(self, mock_chat):
        """generate_with_ollama should use zero temperature."""
        mock_response = {'message': {'content': '{"test": "data"}'}}
        mock_chat.return_value = mock_response
        
        generate_with_ollama("test prompt")
        
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs['options']['temperature'] == 0


class TestExtractQueryAndContext:
    """Test the extract_query_and_context function."""
    
    @patch('graphrag.encoder.generate_with_ollama')
    def test_extract_query_and_context_valid_json(self, mock_generate):
        """extract_query_and_context should parse valid JSON correctly."""
        mock_generate.return_value = '{"cleaned_query": "test query", "extracted_context": "test context"}'
        
        query, context = extract_query_and_context("test prompt")
        
        assert query == "test query"
        assert context == "test context"
    
    @patch('graphrag.encoder.generate_with_ollama')
    def test_extract_query_and_context_none_response(self, mock_generate):
        """extract_query_and_context should raise ValueError on None response."""
        mock_generate.return_value = None
        
        with pytest.raises(ValueError, match="Failed to get a response from Ollama"):
            extract_query_and_context("test prompt")
    
    @patch('graphrag.encoder.generate_with_ollama')
    def test_extract_query_and_context_invalid_json(self, mock_generate):
        """extract_query_and_context should raise ValueError on invalid JSON."""
        mock_generate.return_value = "invalid json"
        
        with pytest.raises(ValueError, match="Failed to decode JSON response"):
            extract_query_and_context("test prompt")
    
    @patch('graphrag.encoder.generate_with_ollama')
    def test_extract_query_and_context_missing_keys(self, mock_generate):
        """extract_query_and_context should raise error if JSON missing required keys."""
        mock_generate.return_value = '{"cleaned_query": "only query"}'
        
        with pytest.raises(KeyError):
            extract_query_and_context("test prompt")
    
    @patch('graphrag.encoder.generate_with_ollama')
    def test_extract_query_and_context_returns_tuple(self, mock_generate):
        """extract_query_and_context should return a tuple of two strings."""
        mock_generate.return_value = '{"cleaned_query": "q", "extracted_context": "c"}'
        
        result = extract_query_and_context("test prompt")
        
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestProcessPdfDocuments:
    """Test the process_pdf_documents async function."""
    
    @pytest.mark.asyncio
    @patch('graphrag.encoder.glob.glob')
    async def test_process_pdf_documents_success(self, mock_glob):
        """process_pdf_documents should call kg_builder.run_async for each PDF."""
        mock_glob.return_value = ["/path/to/doc1.pdf", "/path/to/doc2.pdf"]
        mock_kg_builder = AsyncMock()
        mock_kg_builder.run_async.return_value = {"status": "success"}
        
        await process_pdf_documents(mock_kg_builder)
        
        assert mock_kg_builder.run_async.call_count == 2
    
    @pytest.mark.asyncio
    @patch('graphrag.encoder.glob.glob')
    async def test_process_pdf_documents_empty_list(self, mock_glob):
        """process_pdf_documents should handle empty PDF list."""
        mock_glob.return_value = []
        mock_kg_builder = AsyncMock()
        
        await process_pdf_documents(mock_kg_builder)
        
        mock_kg_builder.run_async.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('graphrag.encoder.glob.glob')
    async def test_process_pdf_documents_calls_with_file_path(self, mock_glob):
        """process_pdf_documents should call run_async with file_path parameter."""
        pdf_path = "/path/to/test.pdf"
        mock_glob.return_value = [pdf_path]
        mock_kg_builder = AsyncMock()
        
        await process_pdf_documents(mock_kg_builder)
        
        mock_kg_builder.run_async.assert_called_once_with(file_path=pdf_path)


class TestProcessTextContext:
    """Test the process_text_context async function."""
    
    @pytest.mark.asyncio
    async def test_process_text_context_success(self):
        """process_text_context should call kg_builder.run_async for each text."""
        mock_kg_builder = AsyncMock()
        mock_kg_builder.run_async.return_value = {"status": "success"}
        texts = ["text1", "text2", "text3"]
        
        await process_text_context(mock_kg_builder, texts)
        
        assert mock_kg_builder.run_async.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_text_context_empty_list(self):
        """process_text_context should handle empty text list."""
        mock_kg_builder = AsyncMock()
        
        await process_text_context(mock_kg_builder, [])
        
        mock_kg_builder.run_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_text_context_calls_with_text_parameter(self):
        """process_text_context should call run_async with text parameter."""
        mock_kg_builder = AsyncMock()
        text = "sample text"
        
        await process_text_context(mock_kg_builder, [text])
        
        mock_kg_builder.run_async.assert_called_once_with(text=text)


class TestProcessContext:
    """Test the process_context function."""
    
    @patch('graphrag.encoder.process_pdf_documents')
    @patch('graphrag.encoder.process_text_context')
    def test_process_context_no_texts(self, mock_text_process, mock_pdf_process):
        """process_context should call PDF processor with no texts."""
        mock_kg_builder_pdf = MagicMock()
        mock_kg_builder_text = MagicMock()
        
        process_context(mock_kg_builder_pdf, mock_kg_builder_text, [])
        
        mock_pdf_process.assert_called_once()
    
    @patch('graphrag.encoder.process_pdf_documents')
    @patch('graphrag.encoder.process_text_context')
    def test_process_context_with_texts(self, mock_text_process, mock_pdf_process):
        """process_context should call both processors when texts provided."""
        mock_kg_builder_pdf = MagicMock()
        mock_kg_builder_text = MagicMock()
        texts = ["text1", "text2"]
        
        process_context(mock_kg_builder_pdf, mock_kg_builder_text, texts)
        
        mock_pdf_process.assert_called_once()
        mock_text_process.assert_called_once()


# ===== INTEGRATION TESTS =====

class TestExtractQueryAndContextIntegration:
    """Integration tests for extract_query_and_context with generate_with_ollama."""
    
    @patch('graphrag.encoder.ollama.chat')
    def test_extract_calls_generate_successfully(self, mock_chat):
        """Test extract_query_and_context successfully calls generate_with_ollama."""
        mock_response = '{"cleaned_query": "What courses?", "extracted_context": "CS student"}'
        mock_chat.return_value = {'message': {'content': mock_response}}
        
        query, context = extract_query_and_context("test prompt")
        
        assert query == "What courses?"
        assert context == "CS student"
        mock_chat.assert_called_once()
    
    @patch('graphrag.encoder.ollama.chat')
    def test_extract_with_complex_json(self, mock_chat):
        """Test extract_query_and_context with more complex context."""
        complex_context = "Complex context with special characters: #$%"
        mock_response = f'{{"cleaned_query": "test query", "extracted_context": "{complex_context}"}}'
        mock_chat.return_value = {'message': {'content': mock_response}}
        
        query, context = extract_query_and_context("test prompt")
        
        assert query == "test query"
        assert context == complex_context


class TestProcessContextIntegration:
    """Integration tests for process_context calling sub-functions."""
    
    @patch('graphrag.encoder.process_text_context', new_callable=AsyncMock)
    @patch('graphrag.encoder.process_pdf_documents', new_callable=AsyncMock)
    def test_process_context_calls_both_processors(self, mock_pdf, mock_text):
        """Test process_context calls both PDF and text processors."""
        mock_kg_pdf = MagicMock()
        mock_kg_text = MagicMock()
        texts = ["sample text"]
        
        process_context(mock_kg_pdf, mock_kg_text, texts)
        
        mock_pdf.assert_called_once_with(mock_kg_pdf)
        mock_text.assert_called_once_with(mock_kg_text, texts)
    
    @patch('graphrag.encoder.process_text_context', new_callable=AsyncMock)
    @patch('graphrag.encoder.process_pdf_documents', new_callable=AsyncMock)
    def test_process_context_passes_correct_builders(self, mock_pdf, mock_text):
        """Test process_context passes correct builders to sub-functions."""
        mock_kg_pdf = MagicMock()
        mock_kg_text = MagicMock()
        
        process_context(mock_kg_pdf, mock_kg_text, [])
        
        mock_pdf.assert_called_with(mock_kg_pdf)
        mock_text.assert_called_with(mock_kg_text, [])
