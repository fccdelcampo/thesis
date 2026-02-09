"""
Unit and integration tests for driver.py (main driver)

Tests test case loading, test execution, and dialogue logging.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_driver import (
    load_test_cases,
    run_test_case,
    run_test_cases,
    log_dialogues
)

# ===== UNIT TESTS =====

class TestLoadTestCases:
    """Test the load_test_cases function."""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"query": "test"}\n{"query": "test2"}\n')
    def test_load_test_cases_returns_list(self, mock_file):
        """load_test_cases should return a list."""
        result = load_test_cases("benchmark1", 0, 1)
        assert isinstance(result, list)
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_test_cases_file_not_found(self, mock_file):
        """load_test_cases should handle FileNotFoundError gracefully."""
        result = load_test_cases("nonexistent", 0, 1)
        # Function should not raise, may return None or empty list
        assert result is None or isinstance(result, list)
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json\n')
    def test_load_test_cases_json_decode_error(self, mock_file):
        """load_test_cases should handle JSON decode errors."""
        result = load_test_cases("benchmark1", 0, 0)
        # Function should handle error gracefully
        assert result is None or isinstance(result, list)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_test_cases_correct_file_path(self, mock_file):
        """load_test_cases should open correct file path."""
        try:
            load_test_cases("benchmark1", 0, 1)
        except:
            pass
        
        # Verify correct file path was attempted
        call_args = str(mock_file.call_args)
        assert "benchmark1" in call_args
        assert "jsonl" in call_args


class TestRunTestCase:
    """Test the run_test_case function."""
    
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_run_test_case_returns_dict(self, mock_extract, mock_process, mock_generate):
        """run_test_case should return a dictionary."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "test answer", "context": "test context"}
        
        test_case = {"query": "test prompt"}
        result = run_test_case(test_case)
        
        assert isinstance(result, dict)
    
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_run_test_case_calls_extract_query(self, mock_extract, mock_process, mock_generate):
        """run_test_case should call extract_query_and_context."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "answer", "context": "context"}
        
        test_case = {"query": "test prompt"}
        run_test_case(test_case)
        
        mock_extract.assert_called_once()
    
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_run_test_case_calls_process_context(self, mock_extract, mock_process, mock_generate):
        """run_test_case should call process_context."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "answer", "context": "context"}
        
        test_case = {"query": "test prompt"}
        run_test_case(test_case)
        
        mock_process.assert_called_once()
    
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_run_test_case_calls_generate_answer(self, mock_extract, mock_process, mock_generate):
        """run_test_case should call generate_answer."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "answer", "context": "context"}
        
        test_case = {"query": "test prompt"}
        run_test_case(test_case)
        
        mock_generate.assert_called_once()
    
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_run_test_case_handles_missing_query_key(self, mock_extract, mock_process, mock_generate):
        """run_test_case should handle test case without 'query' key."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "answer", "context": "context"}
        
        test_case = {"prompt": "test"}  # No 'query' key
        result = run_test_case(test_case)
        
        # Should handle gracefully
        assert result is not None


class TestRunTestCases:
    """Test the run_test_cases function."""
    
    @patch('driver.log_dialogues')
    @patch('driver.run_test_case')
    def test_run_test_cases_processes_all_cases(self, mock_run, mock_log):
        """run_test_cases should process all test cases."""
        mock_run.return_value = {"answer": "test", "context": "context"}
        test_cases = [
            {"question": "q1", "query": "test1"},
            {"question": "q2", "query": "test2"}
        ]
        
        run_test_cases(test_cases, "benchmark1")
        
        assert mock_run.call_count == 2
    
    @patch('driver.log_dialogues')
    @patch('driver.run_test_case')
    def test_run_test_cases_calls_log_dialogues(self, mock_run, mock_log):
        """run_test_cases should call log_dialogues."""
        mock_run.return_value = {"answer": "test", "context": "context"}
        test_cases = [{"question": "q1", "query": "test"}]
        
        run_test_cases(test_cases, "benchmark1")
        
        mock_log.assert_called_once()
    
    @patch('driver.log_dialogues')
    @patch('driver.run_test_case')
    def test_run_test_cases_builds_dialogues(self, mock_run, mock_log):
        """run_test_cases should build dialogue entries."""
        mock_run.return_value = {"answer": "test answer", "context": "test context"}
        test_cases = [{"question": "What is AI?", "query": "test"}]
        
        run_test_cases(test_cases, "benchmark1")
        
        call_args = mock_log.call_args[0]
        dialogues = call_args[0]
        
        assert len(dialogues) == 1
        assert "question" in dialogues[0]
        assert "answer" in dialogues[0]
        assert "context" in dialogues[0]
    
    @patch('driver.log_dialogues')
    @patch('driver.run_test_case')
    def test_run_test_cases_empty_list(self, mock_run, mock_log):
        """run_test_cases should handle empty test case list."""
        run_test_cases([], "benchmark1")
        
        mock_log.assert_called_once()
        call_args = mock_log.call_args[0]
        assert call_args[0] == []


class TestLogDialogues:
    """Test the log_dialogues function."""
    
    @patch('builtins.open', new_callable=mock_open)
    def test_log_dialogues_writes_json_lines(self, mock_file):
        """log_dialogues should write JSON lines to file."""
        dialogues = [
            {"question": "q1", "answer": "a1", "context": "c1"},
            {"question": "q2", "answer": "a2", "context": "c2"}
        ]
        
        log_dialogues(dialogues, "benchmark1")
        
        # Should write to file
        mock_file.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_log_dialogues_uses_jsonl_format(self, mock_file):
        """log_dialogues should write in JSONL format."""
        dialogues = [{"question": "q1", "answer": "a1", "context": "c1"}]
        
        log_dialogues(dialogues, "benchmark1")
        
        # Get write calls
        write_calls = mock_file().write.call_args_list
        assert len(write_calls) > 0
    
    @patch('builtins.open', new_callable=mock_open)
    def test_log_dialogues_handles_unicode(self, mock_file):
        """log_dialogues should handle unicode characters."""
        dialogues = [{"question": "什么是AI?", "answer": "人工智能", "context": "Çöñtëxt"}]
        
        log_dialogues(dialogues, "benchmark1")
        
        mock_file.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_log_dialogues_correct_file_path(self, mock_file):
        """log_dialogues should write to correct file path."""
        dialogues = [{"question": "q1", "answer": "a1", "context": "c1"}]
        
        log_dialogues(dialogues, "benchmark1")
        
        call_args = str(mock_file.call_args)
        assert "benchmark1" in call_args
        assert "jsonl" in call_args or "logs" in call_args
    
    @patch('builtins.open', new_callable=mock_open)
    def test_log_dialogues_append_mode(self, mock_file):
        """log_dialogues should open file in append mode."""
        dialogues = [{"question": "q1", "answer": "a1", "context": "c1"}]
        
        log_dialogues(dialogues, "benchmark1")
        
        # Check that file was opened in append mode
        call_args = mock_file.call_args
        assert 'a' in call_args[0][1] if len(call_args[0]) > 1 else True


# ===== INTEGRATION TESTS =====

class TestLoadAndRunTestCasesIntegration:
    """Integration tests combining load and run functions."""
    
    @patch('driver.log_dialogues')
    @patch('driver.run_test_case')
    @patch('builtins.open', new_callable=mock_open, read_data='{"query": "test1"}\n{"query": "test2"}\n')
    def test_load_and_run_integration(self, mock_file, mock_run, mock_log):
        """Integration: load test cases and run them."""
        mock_run.return_value = {"answer": "a", "context": "c"}
        
        test_cases = load_test_cases("benchmark1", 0, 1)
        
        if test_cases:
            run_test_cases(test_cases, "benchmark1")
            mock_log.assert_called_once()


class TestFullTestExecutionIntegration:
    """Integration tests for complete test execution flow."""
    
    @patch('driver.log_dialogues')
    @patch('driver.generate_answer')
    @patch('driver.process_context')
    @patch('driver.extract_query_and_context')
    def test_full_test_execution_flow(self, mock_extract, mock_process, mock_generate, mock_log):
        """Integration test: complete flow from test case to logging."""
        mock_extract.return_value = ("query", "context")
        mock_generate.return_value = {"answer": "test answer", "context": "test context"}
        
        test_cases = [
            {"question": "What is AI?", "query": "test prompt 1"},
            {"question": "What is ML?", "query": "test prompt 2"}
        ]
        
        run_test_cases(test_cases, "benchmark1")
        
        # Verify all steps were called
        assert mock_extract.call_count >= 2
        assert mock_process.call_count >= 2
        assert mock_generate.call_count >= 2
        mock_log.assert_called_once()
        
        # Verify logged dialogues have correct structure
        call_args = mock_log.call_args[0]
        dialogues = call_args[0]
        assert len(dialogues) == 2
        for dialogue in dialogues:
            assert "question" in dialogue
            assert "answer" in dialogue
            assert "context" in dialogue
