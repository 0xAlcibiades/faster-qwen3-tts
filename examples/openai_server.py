#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Supports three generation modes:
  - voice_clone : reference-audio-based cloning (default, backward-compatible)
  - custom_voice: predefined speaker IDs with optional instruction
  - voice_design : instruction-based voice synthesis

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Voice cloning (default):
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # CustomVoice model:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --port 8000

    # VoiceDesign model:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \\
        --port 8000

API usage — voice_clone (default):
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav

API usage — voice_design (instruction-based):
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "tts-1",
            "input": "Hello world.",
            "instructions": "Warm, confident British narrator"
        }' \\
        --output speech.wav

API usage — custom_voice (speaker ID):
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "tts-1",
            "input": "Hello world.",
            "voice": "aiden",
            "response_format": "wav"
        }' \\
        --output speech.wav
"""
import argparse
import asyncio
import io
import json
import logging
import os
import queue
import struct
import sys
import threading
from typing import Any, AsyncGenerator, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
model_type: Optional[str] = None  # "base", "custom_voice", or "voice_design"
_model_lock = threading.Lock()  # prevent concurrent GPU inference

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CustomVoiceObject(BaseModel):
    """Represents a custom voice object as per OpenAI API spec."""
    id: str
    name: Optional[str] = None
    preview_url: Optional[str] = None


_VALID_MODES = frozenset(["voice_clone", "custom_voice", "voice_design"])


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: Union[str, dict, CustomVoiceObject] = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    # NEW — instruction-based voice control (VoiceDesign mode)
    instructions: Optional[str] = None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def _get_voice_name(req: SpeechRequest) -> str:
    """Extract the voice name from a SpeechRequest, handling str/dict/CustomVoiceObject."""
    if isinstance(req.voice, str):
        return req.voice
    elif isinstance(req.voice, dict):
        return req.voice.get("voice_name", req.voice.get("id", "alloy"))
    elif isinstance(req.voice, CustomVoiceObject):
        return req.voice.id
    return "alloy"


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


def _resolve_mode(req: SpeechRequest) -> str:
    """Resolve the generation mode from an explicit request or infer from shape."""
    # Infer from request shape
    if req.instructions and isinstance(req.voice, str):
        return "voice_design"
    if isinstance(req.voice, dict):
        return "custom_voice"
    if isinstance(req.voice, CustomVoiceObject):
        return "custom_voice"
    return "voice_clone"


def _validate_mode_compatibility(mode: str) -> None:
    """Raise 400 if the requested mode is incompatible with the loaded model."""
    if model_type is None:
        return  # model not loaded yet — will be caught downstream

    compat = {
        "voice_clone":    {"base", "custom_voice", "voice_design"},
        "custom_voice":   {"custom_voice"},
        "voice_design":   {"voice_design"},
    }

    allowed_modes = compat.get(mode, set())
    if model_type not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Mode {mode!r} requires a {model_type}-variant model "
                f"(loaded: {model_type}). Allowed modes for this model: "
                f"{sorted(allowed_modes)}"
            ),
        )


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(req: SpeechRequest, mode: str) -> AsyncGenerator[bytes, None]:
    """
    Run the appropriate generate_*_streaming method in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                if mode == "voice_clone":
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                        text=req.input,
                        language=voice_cfg.get("language", "Auto"),
                        ref_audio=voice_cfg["ref_audio"],
                        ref_text=voice_cfg.get("ref_text", ""),
                        chunk_size=12,
                        non_streaming_mode=False,
                    ):
                        q.put(chunk)
                elif mode == "custom_voice":
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    for chunk, _sr, _timing in tts_model.generate_custom_voice_streaming(
                        text=req.input,
                        speaker=voice_cfg.get("speaker", "alloy"),
                        language=voice_cfg.get("language", "Auto"),
                        instruct=req.instructions,
                        chunk_size=12,
                    ):
                        q.put(chunk)
                else:  # voice_design
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    for chunk, _sr, _timing in tts_model.generate_voice_design_streaming(
                        text=req.input,
                        instruct=req.instructions or "",
                        language=voice_cfg.get("language", "Auto"),
                        chunk_size=12,
                    ):
                        q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    mode = _resolve_mode(req)
    _validate_mode_compatibility(mode)

    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                if mode == "voice_clone":
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    return tts_model.generate_voice_clone(
                        text=req.input,
                        language=voice_cfg.get("language", "Auto"),
                        ref_audio=voice_cfg["ref_audio"],
                        ref_text=voice_cfg.get("ref_text", ""),
                    )
                elif mode == "custom_voice":
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    return tts_model.generate_custom_voice(
                        text=req.input,
                        speaker=voice_cfg.get("speaker", "alloy"),
                        language=voice_cfg.get("language", "Auto"),
                        instruct=req.instructions,
                    )
                else:
                    voice_cfg = resolve_voice(_get_voice_name(req))
                    return tts_model.generate_voice_design(
                        text=req.input,
                        instruct=req.instructions or "",
                        language=voice_cfg.get("language", "Auto"),
                    )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(req, mode):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE, model_type

    args = _parse_args()

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        print(
            "ERROR: provide --ref-audio <file> or --voices <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    model_type = tts_model.model.model.tts_model_type
    logger.info("Model ready. Sample rate: %d Hz, model type: %s", SAMPLE_RATE, model_type)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
