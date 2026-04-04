"""Tests for protor.llm_backends module."""

import os
import pytest
from unittest.mock import patch, MagicMock
from protor.llm_backends import (
    LLMBackend,
    OllamaBackend,
    create_backend,
    BACKEND_CHOICES,
)


class TestBackendChoices:
    def test_choices_not_empty(self):
        assert len(BACKEND_CHOICES) == 3

    def test_expected_backends(self):
        assert "ollama" in BACKEND_CHOICES
        assert "openai" in BACKEND_CHOICES
        assert "anthropic" in BACKEND_CHOICES


class TestOllamaBackendInit:
    def test_default_base_url(self):
        backend = OllamaBackend("llama3")
        assert backend._model == "llama3"
        assert backend._base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        backend = OllamaBackend("mistral", base_url="http://custom:11434")
        assert backend._base_url == "http://custom:11434"

    def test_env_base_url(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:9999"}):
            backend = OllamaBackend("llama3")
            assert backend._base_url == "http://env-host:9999"

    def test_model_name_property(self):
        backend = OllamaBackend("llama3")
        assert backend.model_name == "llama3"


class TestOllamaBackendCheckAvailable:
    @patch("requests.get")
    def test_available(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        backend = OllamaBackend("llama3")
        assert backend.check_available() is True

    @patch("requests.get")
    def test_not_available(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        backend = OllamaBackend("llama3")
        assert backend.check_available() is False

    @patch("requests.get")
    def test_not_available_500(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        backend = OllamaBackend("llama3")
        assert backend.check_available() is False


class TestOllamaBackendStream:
    @patch("rich.console.Console")
    @patch("requests.post")
    def test_stream_success(self, mock_post, mock_console_cls):
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        lines = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": true}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        backend = OllamaBackend("llama3")
        result = backend.stream("Test prompt")

        assert result == "Hello world"
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": "Test prompt", "stream": True},
            stream=True,
            timeout=300,
        )

    @patch("rich.console.Console")
    @patch("requests.post")
    def test_stream_model_not_found(self, mock_post, mock_console_cls):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        backend = OllamaBackend("nonexistent")
        with pytest.raises(RuntimeError, match="not found"):
            backend.stream("Test prompt")

    @patch("rich.console.Console")
    @patch("requests.post")
    def test_stream_handles_empty_lines(self, mock_post, mock_console_cls):
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        lines = [
            b"",
            b'{"response": "Hi", "done": true}',
            b"",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        backend = OllamaBackend("llama3")
        result = backend.stream("Test")

        assert result == "Hi"


class TestCreateBackend:
    def test_create_ollama(self):
        backend = create_backend("ollama", "llama3")
        assert isinstance(backend, OllamaBackend)
        assert backend.model_name == "llama3"

    def test_create_ollama_with_base_url(self):
        backend = create_backend("ollama", "mistral", base_url="http://custom:11434")
        assert backend._base_url == "http://custom:11434"

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown", "model")

    def test_create_backend_case_insensitive(self):
        backend = create_backend("OLLAMA", "llama3")
        assert isinstance(backend, OllamaBackend)


class TestOpenAIBackend:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                from protor.llm_backends import OpenAIBackend

                OpenAIBackend("gpt-4")

    def test_api_key_from_param(self):
        from protor.llm_backends import OpenAIBackend

        backend = OpenAIBackend("gpt-4", api_key="test-key")
        assert backend._api_key == "test-key"
        assert backend.model_name == "gpt-4"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_api_key_from_env(self):
        from protor.llm_backends import OpenAIBackend

        backend = OpenAIBackend("gpt-4")
        assert backend._api_key == "test-key"


class TestAnthropicBackend:
    def test_missing_api_key_raises(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            from protor.llm_backends import AnthropicBackend

            AnthropicBackend("claude-3")

    def test_api_key_from_param(self):
        from protor.llm_backends import AnthropicBackend

        backend = AnthropicBackend("claude-3", api_key="test-key")
        assert backend._api_key == "test-key"
        assert backend.model_name == "claude-3"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_api_key_from_env(self):
        from protor.llm_backends import AnthropicBackend

        backend = AnthropicBackend("claude-3")
        assert backend._api_key == "test-key"
