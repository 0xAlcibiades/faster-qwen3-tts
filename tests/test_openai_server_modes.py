"""Tests for OpenAI-compatible server mode resolution, validation, and dispatch."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


@pytest.fixture
def mock_model():
    """Create a mock FasterQwen3TTS model with all generation methods."""
    model = MagicMock()
    model.generate_voice_clone_streaming.return_value = iter([])
    model.generate_custom_voice_streaming.return_value = iter([])
    model.generate_voice_design_streaming.return_value = iter([])
    model.generate_voice_clone.return_value = ([MagicMock()], 24000)
    model.generate_custom_voice.return_value = ([MagicMock()], 24000)
    model.generate_voice_design.return_value = ([MagicMock()], 24000)
    model.model = MagicMock()
    model.model.tts_model_type = "base"
    return model


class TestModeResolution:
    """Test _resolve_mode function."""

    def test_infer_voice_design_from_instructions(self):
        from examples.openai_server import SpeechRequest, _resolve_mode

        req = SpeechRequest(input="hello", instructions="Warm British narrator")
        assert _resolve_mode(req) == "voice_design"

    def test_infer_custom_voice_from_dict_voice(self):
        from examples.openai_server import SpeechRequest, _resolve_mode

        req = SpeechRequest(input="hello", voice={"id": "custom_voice_1"})
        assert _resolve_mode(req) == "custom_voice"

    def test_infer_custom_voice_from_custom_voice_object(self):
        from examples.openai_server import CustomVoiceObject, SpeechRequest, _resolve_mode

        req = SpeechRequest(input="hello", voice=CustomVoiceObject(id="custom_voice_1"))
        assert _resolve_mode(req) == "custom_voice"

    def test_infer_voice_clone_by_default(self):
        from examples.openai_server import SpeechRequest, _resolve_mode

        req = SpeechRequest(input="hello", voice="alloy")
        assert _resolve_mode(req) == "voice_clone"

    def test_instructions_override_voice_config(self):
        from examples.openai_server import SpeechRequest, _resolve_mode

        # When instructions are present, voice_design takes precedence
        req = SpeechRequest(input="hello", instructions="Warm voice", voice="alloy")
        assert _resolve_mode(req) == "voice_design"

    def test_dict_voice_overrides_instructions(self):
        from examples.openai_server import SpeechRequest, _resolve_mode

        # Custom voice object takes precedence over instructions
        req = SpeechRequest(
            input="hello",
            instructions="Warm voice",
            voice={"id": "custom_voice_1"}
        )
        assert _resolve_mode(req) == "custom_voice"


class TestModeValidation:
    """Test _validate_mode_compatibility function."""

    def test_voice_clone_compatible_with_base(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", "base"):
            _validate_mode_compatibility("voice_clone")  # Should not raise

    def test_voice_clone_compatible_with_custom_voice_model(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", "custom_voice"):
            _validate_mode_compatibility("voice_clone")  # Should not raise

    def test_voice_clone_compatible_with_voice_design_model(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", "voice_design"):
            _validate_mode_compatibility("voice_clone")  # Should not raise

    def test_custom_voice_requires_custom_voice_model(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", "base"):
            with pytest.raises(Exception) as exc_info:
                _validate_mode_compatibility("custom_voice")
            assert "requires" in str(exc_info.value).lower()

    def test_voice_design_requires_voice_design_model(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", "base"):
            with pytest.raises(Exception) as exc_info:
                _validate_mode_compatibility("voice_design")
            assert "requires" in str(exc_info.value).lower()

    def test_unknown_model_type_allows_voice_clone(self):
        from examples.openai_server import _validate_mode_compatibility

        with patch("examples.openai_server.model_type", None):
            _validate_mode_compatibility("voice_clone")  # Should not raise


class TestEndpointIntegration:
    """Test the full endpoint with different modes."""

    def test_voice_clone_streaming_works(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            with patch("examples.openai_server.model_type", "base"):
                mock_model.generate_voice_clone_streaming.return_value = iter([])
                # Configure a mock voice
                mock_voices = {"alloy": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English"}}
                with patch("examples.openai_server.voices", mock_voices):
                    client = TestClient(app)
                    resp = client.post(
                        "/v1/audio/speech",
                        json={
                            "input": "hello",
                            "voice": "alloy",
                        },
                    )
                    assert resp.status_code == 200

    def test_custom_voice_routes_correctly(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            with patch("examples.openai_server.model_type", "custom_voice"):
                mock_model.generate_custom_voice_streaming.return_value = iter([])
                # Configure a mock voice with speaker
                mock_voices = {"aiden": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English", "speaker": "aiden"}}
                with patch("examples.openai_server.voices", mock_voices):
                    client = TestClient(app)
                    resp = client.post(
                        "/v1/audio/speech",
                        json={
                            "input": "hello",
                            "voice": {"id": "aiden"},
                        },
                    )
                    assert resp.status_code == 200

    def test_voice_design_routes_correctly(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            with patch("examples.openai_server.model_type", "voice_design"):
                mock_model.generate_voice_design_streaming.return_value = iter([])
                # Configure a mock voice
                mock_voices = {"alloy": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English"}}
                with patch("examples.openai_server.voices", mock_voices):
                    client = TestClient(app)
                    resp = client.post(
                        "/v1/audio/speech",
                        json={
                            "input": "hello",
                            "instructions": "Warm British narrator",
                        },
                    )
                    assert resp.status_code == 200

    def test_incompatible_mode_returns_400(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            with patch("examples.openai_server.model_type", "base"):
                client = TestClient(app)
                resp = client.post(
                    "/v1/audio/speech",
                    json={
                        "input": "hello",
                        "instructions": "Warm British narrator",
                    },
                )
                assert resp.status_code == 400

    def test_backward_compat_no_instructions(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            with patch("examples.openai_server.model_type", "base"):
                mock_model.generate_voice_clone_streaming.return_value = iter([])
                # Configure a mock voice
                mock_voices = {"alloy": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English"}}
                with patch("examples.openai_server.voices", mock_voices):
                    client = TestClient(app)
                    resp = client.post(
                        "/v1/audio/speech",
                        json={
                            "input": "hello",
                            "voice": "alloy",
                        },
                    )
                    assert resp.status_code == 200

    def test_empty_input_returns_400(self, mock_model):
        from examples.openai_server import app

        with patch("examples.openai_server.tts_model", mock_model):
            client = TestClient(app)
            resp = client.post(
                "/v1/audio/speech",
                json={
                    "input": "",
                    "voice": "alloy",
                },
            )
            assert resp.status_code == 400


class TestVoiceConfigFallback:
    """Test that voice config resolution still works for voice_clone mode."""

    def test_voice_lookup_in_voices_dict(self):
        from examples.openai_server import resolve_voice, voices

        mock_voices = {"alloy": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English"}}
        with patch("examples.openai_server.voices", mock_voices):
            cfg = resolve_voice("alloy")
            assert cfg["ref_audio"] == "voice.wav"

    def test_fallback_to_default_voice(self):
        from examples.openai_server import resolve_voice, default_voice, voices

        mock_voices = {"default": {"ref_audio": "voice.wav", "ref_text": "test", "language": "English"}}
        with patch("examples.openai_server.voices", mock_voices):
            with patch("examples.openai_server.default_voice", "default"):
                cfg = resolve_voice("unknown_voice")
                assert cfg["ref_audio"] == "voice.wav"
