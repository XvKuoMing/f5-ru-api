from __future__ import annotations

import base64
from typing import Any, Dict, Generator, List, Optional

import numpy as np
from fastapi import Body, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from tts.config import get_settings
from tts.engine import get_engine

app = FastAPI(title="OpenAI-compatible TTS", version="0.1.0")


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
	s = get_settings()
	return {
		"object": "list",
		"data": [
			{
				"id": s.model_repo.split("/")[-1],
				"object": "model",
				"owned_by": "local",
			}
		],
	}


@app.get("/v1/voices")
def list_voices() -> Dict[str, Any]:
	engine = get_engine()
	return {"object": "list", "data": engine.voices}


@app.post("/v1/audio/speech")
def tts_speech(
	payload: Dict[str, Any] = Body(...),
	accept: Optional[str] = Header(default=None, alias="Accept"),
):
	engine = get_engine()
	settings = get_settings()

	text: str = payload.get("input") or payload.get("text")
	if not text or not isinstance(text, str):
		raise HTTPException(status_code=400, detail="Field 'input' (text) is required")
	voice: Optional[str] = payload.get("voice")
	response_format: str = (payload.get("response_format") or "wav").lower()
	speed: Optional[float] = payload.get("speed")
	stream: bool = bool(payload.get("stream") or False)

	if stream:
		def gen() -> Generator[bytes, None, None]:
			for chunk in engine.synthesize_stream_pcm16(text, voice_name=voice, speed=speed):
				yield chunk
		headers = {
			"X-Content-Type": "audio/pcm; codec=pcm_s16le",
			"X-Audio-Sample-Rate": str(settings.target_sample_rate),
		}
		return StreamingResponse(gen(), headers=headers, media_type="application/octet-stream")

	audio, sr = engine.synthesize_numpy(text, voice_name=voice, speed=speed)
	if response_format == "wav":
		wav_bytes = engine.encode_wav(audio, sr)
		return Response(content=wav_bytes, media_type="audio/wav")
	elif response_format in {"pcm", "pcm16", "s16le"}:
		pcm16 = np.clip(audio, -1.0, 1.0)
		pcm16 = (pcm16 * 32767.0).astype(np.int16)
		return Response(content=pcm16.tobytes(), media_type="application/octet-stream")
	elif response_format in {"base64", "b64"}:
		wav_bytes = engine.encode_wav(audio, sr)
		return JSONResponse({"b64_audio": base64.b64encode(wav_bytes).decode("utf-8"), "sample_rate": sr})
	else:
		raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")


@app.post("/v1/audio/speech/batch")
def tts_batch(payload: Dict[str, Any] = Body(...)):
	items: List[Dict[str, Any]] = payload.get("items") or []
	if not isinstance(items, list) or not items:
		raise HTTPException(status_code=400, detail="'items' must be a non-empty list")
	engine = get_engine()
	out: List[Dict[str, Any]] = []
	for item in items:
		text: str = item.get("input") or item.get("text")
		voice: Optional[str] = item.get("voice")
		response_format: str = (item.get("response_format") or "base64").lower()
		audio, sr = engine.synthesize_numpy(text, voice_name=voice)
		wav_bytes = engine.encode_wav(audio, sr)
		out.append({"voice": voice, "b64_audio": base64.b64encode(wav_bytes).decode("utf-8"), "sample_rate": sr, "format": response_format})
	return {"object": "list", "data": out}


@app.get("/healthz")
def health() -> Dict[str, str]:
	engine = get_engine()
	return {"status": "ok", "device": engine.device}
