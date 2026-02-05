from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from fastapi import Query
from typing import Literal, Annotated
from f5 import (
	F5Settings, 
	AccentSettings,
	F5EnginePool
	)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from contextlib import asynccontextmanager
import logging
from models.voice import Voices, Voice
from models.model import Models, Model
from models.generation import AudioSpeechRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvSettings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", extra="ignore")
	host: str = Field(default="0.0.0.0", env="HOST")
	port: int = Field(default=8000, env="PORT")
	model_repo: str = Field(default="ESpeech/ESpeech-TTS-1_RL-V2", env="MODEL_REPO")
	model_file: str = Field(default="espeech_tts_rlv2.pt", env="MODEL_FILE")
	vocab_file: str = Field(default="vocab.txt", env="VOCAB_FILE")
	voices_dir: str = Field(default="./voices", env="VOICES_DIR")
	instances: int = Field(default=1, env="INSTANCES")


settings = EnvSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
	f5_settings = F5Settings(
		model_repo=settings.model_repo,
		model_file=settings.model_file,
		vocab_file=settings.vocab_file,
	)
	accent_settings = AccentSettings()
	app.state.f5_engine = F5EnginePool(
		f5_settings,
		accent_settings,
		settings.voices_dir,
		num_instances=settings.instances,
	)
	yield
	app.state.f5_engine = None


app = FastAPI(lifespan=lifespan)
F5Dep = Annotated[F5EnginePool, Depends(lambda: app.state.f5_engine)]


@app.get("/voices")
async def get_voices(f5_engine: F5Dep) -> Voices:
	return Voices(voices=[Voice(name=voice) for voice in f5_engine.voices])

@app.get("/models")
async def get_model(f5_engine: F5Dep) -> Models:
	return Models(models=[Model(name=f5_engine.model_name)])

@app.post(
	"/v1/audio/speech",
	response_model=None,
	responses={
		200: {
			"content": {
				"audio/wav": {},
				"audio/pcm": {},
			}
		}
	}
)
async def generate_audio_speech(request: AudioSpeechRequest, f5_engine: F5Dep) -> Response | StreamingResponse:
	if request.model != f5_engine.model_name:
		raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
	if request.voice not in f5_engine.voices:
		raise HTTPException(status_code=400, detail=f"Voice {request.voice} not found")
	if request.response_format not in ["pcm", "wav"]:
		raise HTTPException(status_code=400, detail=f"Format {request.response_format} not supported")
	# Stream if requested (only supports pcm16)
	if getattr(request, "stream", False):
		if request.response_format != "pcm":
			raise HTTPException(status_code=400, detail="Streaming only supports pcm format")
		gen, sample_rate = f5_engine.generate_stream(voice=request.voice, gen_text=request.input, response_format=request.response_format)
		return StreamingResponse(iterate_in_threadpool(gen), media_type="audio/pcm", headers={
			"x-audio-sample-rate": str(sample_rate),
		})
	# Non-streaming: use pool (async) to run generation
	audio_bytes, sample_rate, duration = await f5_engine.generate(
		voice=request.voice,
		gen_text=request.input,
		response_format=request.response_format,
	)
	media_type = "audio/wav" if request.response_format == "wav" else "audio/pcm"
	headers = {
		"x-audio-sample-rate": str(sample_rate),
		"x-audio-duration-seconds": f"{duration:.6f}",
	}
	return Response(content=audio_bytes, media_type=media_type, headers=headers)

@app.post(
	"/v1/audio/speech/stream",
	responses={
		200: {
			"content": {
				"audio/pcm": {},
			}
		}
	}
)
async def generate_audio_speech_stream(request: AudioSpeechRequest, f5_engine: F5Dep) -> StreamingResponse:
	if request.model != f5_engine.model_name and request.model not in ["tts-1", "tts", "f5"]:
		raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
	if request.voice not in f5_engine.voices:
		raise HTTPException(status_code=400, detail=f"Voice {request.voice} not found")
	if request.response_format != "pcm":
		raise HTTPException(status_code=400, detail=f"Streaming only supports pcm format")
	gen, sample_rate = await f5_engine.start_stream(voice=request.voice, gen_text=request.input, response_format=request.response_format)
	return StreamingResponse(iterate_in_threadpool(gen), media_type="audio/pcm", headers={
		"x-audio-sample-rate": str(sample_rate),
	})

@app.get(
	"/v1/audio/speech",
	responses={
		200: {
			"content": {
				"audio/wav": {},
				"audio/pcm": {},
			}
		}
	}
)
async def generate_audio_speech_get(
	f5_engine: F5Dep,
	input: str = Query(..., description="Input text"),
	voice: str = Query(..., description="Voice name"),
	model: str = Query(..., description="Model name"),
	response_format: Literal["pcm", "wav"] = Query("wav", description="Response format")
) -> Response:
	if model != f5_engine.model_name:
		raise HTTPException(status_code=400, detail=f"Model {model} not found")
	if voice not in f5_engine.voices:
		raise HTTPException(status_code=400, detail=f"Voice {voice} not found")
	if response_format not in ["pcm", "wav"]:
		raise HTTPException(status_code=400, detail=f"Format {response_format} not supported")
	audio_bytes, sample_rate, duration = await f5_engine.generate(
		voice=voice,
		gen_text=input,
		response_format=response_format,
	)
	media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
	headers = {
		"x-audio-sample-rate": str(sample_rate),
		"x-audio-duration-seconds": f"{duration:.6f}",
	}
	return Response(content=audio_bytes, media_type=media_type, headers=headers)

if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host=settings.host, port=settings.port)






