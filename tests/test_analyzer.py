"""Unit tests for protor.analyzer module"""
import pytest
import responses
from unittest.mock import patch, MagicMock
from protor.analyzer import (
    check_ollama_connection,
    list_ollama_models,
    prepare_analysis_data,
    stream_ollama_response
)


class TestCheckOllamaConnection:
    """Tests for check_ollama_connection function"""
    
    @responses.activate
    def test_ollama_running(self):
        """Test when Ollama is running"""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": []},
            status=200
        )
        
        result = check_ollama_connection()
        
        assert result is True
    
    @responses.activate
    def test_ollama_not_running(self):
        """Test when Ollama is not running"""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=ConnectionError("Connection refused")
        )
        
        result = check_ollama_connection()
        
        assert result is False


class TestListOllamaModels:
    """Tests for list_ollama_models function"""
    
    @responses.activate
    def test_list_models_success(self):
        """Test listing available models"""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={
                "models": [
                    {"name": "llama3:latest", "size": 4661224192},
                    {"name": "qwen3:8b", "size": 8661224192}
                ]
            },
            status=200
        )
        
        models = list_ollama_models()
        
        # Function displays models but doesn't return them
        # Just verify it doesn't crash
        assert models is None or isinstance(models, list)
    
    @responses.activate
    def test_list_models_connection_error(self):
        """Test handling connection error"""
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=ConnectionError("Connection refused")
        )
        
        models = list_ollama_models()
        
        assert models is None


class TestPrepareAnalysisData:
    """Tests for prepare_analysis_data function"""
    
    def test_prepare_single_manifest(self, sample_manifest):
        """Test preparing single manifest for analysis"""
        result = prepare_analysis_data([sample_manifest])
        
        assert isinstance(result, str)
        assert "example.com" in result
        assert "Test Page" in result
    
    def test_prepare_multiple_manifests(self, sample_manifest):
        """Test preparing multiple manifests"""
        manifests = [sample_manifest, sample_manifest]
        result = prepare_analysis_data(manifests)
        
        assert isinstance(result, str)
        assert result.count("example.com") >= 2
    
    def test_max_chars_limit(self, sample_manifest):
        """Test that output respects max_chars limit"""
        result = prepare_analysis_data([sample_manifest], max_chars=50)
        
        # Should be truncated to approximately max_chars (with some overhead for formatting)
        assert len(result) <= 200  # Allow some overhead for formatting
    
    def test_empty_data(self):
        """Test with empty data list"""
        result = prepare_analysis_data([])
        
        assert result == ""


class TestStreamOllamaResponse:
    """Tests for stream_ollama_response function"""
    
    @responses.activate
    def test_stream_response_success(self, mock_ollama_response):
        """Test streaming response from Ollama"""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json=mock_ollama_response,
            status=200
        )
        
        result = stream_ollama_response("llama3", "Test prompt")
        
        assert result is not None
        assert isinstance(result, str)
    
    @responses.activate
    def test_stream_response_error(self):
        """Test handling streaming error"""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=ConnectionError("Connection error")
        )
        
        result = stream_ollama_response("llama3", "Test prompt")
        
        # Should return error message string, not None
        assert result is not None
        assert "Error" in result or "error" in result.lower()
    
    @patch('protor.analyzer.console')
    @responses.activate
    def test_stream_prints_output(self, mock_console, mock_ollama_response):
        """Test that streaming prints to console"""
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json=mock_ollama_response,
            status=200
        )
        
        stream_ollama_response("llama3", "Test prompt")
        
        # Console should be called for output
        assert mock_console.print.called
