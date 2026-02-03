"""
Unit tests for graphrag/config.py

Tests configuration constants and their values.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphrag.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    PROMPT_TEMPLATE,
    RETRIEVAL_QUERY
)


class TestConfigConstants:
    """Test that configuration constants are properly defined."""
    
    def test_neo4j_uri_is_string(self):
        """NEO4J_URI should be a string."""
        assert isinstance(NEO4J_URI, str)
        assert len(NEO4J_URI) > 0
    
    def test_neo4j_uri_format(self):
        """NEO4J_URI should have valid neo4j protocol format."""
        assert NEO4J_URI.startswith("neo4j://")
    
    def test_neo4j_username_is_string(self):
        """NEO4J_USERNAME should be a string."""
        assert isinstance(NEO4J_USERNAME, str)
        assert len(NEO4J_USERNAME) > 0
    
    def test_neo4j_password_is_string(self):
        """NEO4J_PASSWORD should be a string."""
        assert isinstance(NEO4J_PASSWORD, str)
        assert len(NEO4J_PASSWORD) > 0
    
    def test_prompt_template_is_string(self):
        """PROMPT_TEMPLATE should be a string."""
        assert isinstance(PROMPT_TEMPLATE, str)
        assert len(PROMPT_TEMPLATE) > 0
    
    def test_prompt_template_contains_placeholders(self):
        """PROMPT_TEMPLATE should contain required placeholders."""
        assert "{" in PROMPT_TEMPLATE and "}" in PROMPT_TEMPLATE
    
    def test_retrieval_query_is_string(self):
        """RETRIEVAL_QUERY should be a string."""
        assert isinstance(RETRIEVAL_QUERY, str)
        assert len(RETRIEVAL_QUERY) > 0
    
    def test_retrieval_query_is_cypher_like(self):
        """RETRIEVAL_QUERY should contain MATCH keyword for Cypher."""
        assert "MATCH" in RETRIEVAL_QUERY or "match" in RETRIEVAL_QUERY.upper()
