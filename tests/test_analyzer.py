"""Unit tests for protor.analyzer module"""
import json
import pytest
import responses
from unittest.mock import patch, MagicMock
from protor.analyzer import (
    check_ollama,
    list_ollama_models,
    _prepare_context,
    _stream,
    analyze_with_ollama,
    FOCUS_CHOICES,
)
from protor.exceptions import OllamaModelNotFoundError, OllamaUnavailableError
from protor.models import SiteManifest, SiteMetadata, AnalysisResult


class TestCheckOllama:
    @responses.activate
    def test_ollama_running(self):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": []},
            status=200,
        )
        assert check_ollama() is True

    @responses.activate
    def test_ollama_not_running(self):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=ConnectionError("Connection refused"),
        )
        assert check_ollama() is False

    def test_custom_base_url(self):
        assert check_ollama("http://invalid-host:9999") is False


class TestListOllamaModels:
    @responses.activate
    @patch("protor.analyzer.console")
    def test_list_models_success(self, mock_console):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={
                "models": [
                    {"name": "llama3:latest", "size": 4661224192, "modified_at": "2024-01-01T00:00:00Z"},
                ]
            },
            status=200,
        )
        list_ollama_models()
        assert mock_console.print.called

    @responses.activate
    @patch("protor.analyzer.console")
    def test_list_models_empty(self, mock_console):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": []},
            status=200,
        )
        list_ollama_models()
        assert mock_console.print.called

    @responses.activate
    @patch("protor.analyzer.console")
    def test_list_models_ollama_unavailable(self, mock_console):
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=ConnectionError("Connection refused"),
        )
        list_ollama_models()
        assert mock_console.print.called


class TestPrepareContext:
    def test_prepare_single_manifest(self, sample_manifest):
        result = _prepare_context([sample_manifest])
        assert isinstance(result, str)
        assert "example.com" in result
        assert "Example Domain" in result

    def test_prepare_multiple_manifests(self, sample_manifest):
        manifests = [sample_manifest, sample_manifest]
        result = _prepare_context(manifests)
        assert isinstance(result, str)
        assert result.count("example.com") >= 2

    def test_max_chars_limit(self, sample_manifest):
        result = _prepare_context([sample_manifest])
        assert len(result) <= 8000 + 20

    def test_empty_data(self):
        result = _prepare_context([])
        assert result == ""

    def test_dict_input(self):
        data = [{
            "domain": "test.com",
            "url": "https://test.com",
            "metadata": {"title": "Test", "description": "Desc"},
            "js_count": 3,
            "text_content": "Hello world",
        }]
        result = _prepare_context(data)
        assert "test.com" in result
        assert "Hello world" in result


class TestStream:
    @responses.activate
    def test_stream_response_success(self, mock_ollama_response):
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={**mock_ollama_response, "done": True},
            status=200,
        )
        result = _stream("llama3", "Test prompt")
        assert isinstance(result, str)

    @responses.activate
    def test_stream_model_not_found(self):
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            status=404,
        )
        with pytest.raises(OllamaModelNotFoundError):
            _stream("nonexistent", "Test prompt")


class TestFocusChoices:
    def test_focus_choices_not_empty(self):
        assert len(FOCUS_CHOICES) > 0

    def test_expected_focus_modes(self):
        assert "general" in FOCUS_CHOICES
        assert "technical" in FOCUS_CHOICES
        assert "content" in FOCUS_CHOICES
        assert "seo" in FOCUS_CHOICES
