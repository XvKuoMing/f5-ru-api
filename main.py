from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import Response
from fastapi import Query
from typing import Literal
from f5 import (
	F5Engine, 
	F5Settings, 
	AccentSettings
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
	model_config = SettingsConfigDict(env_file=".env")
	host: str = Field(default="0.0.0.0", env="HOST")
	port: int = Field(default=8000, env="PORT")
	model_repo: str = Field(default="ESpeech/ESpeech-TTS-1_RL-V2", env="MODEL_REPO")
	model_file: str = Field(default="espeech_tts_rlv2.pt", env="MODEL_FILE")
	vocab_file: str = Field(default="vocab.txt", env="VOCAB_FILE")
	voices_dir: str = Field(default="./voices", env="VOICES_DIR")


settings = EnvSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
	f5_settings = F5Settings(
		model_repo=settings.model_repo,
		model_file=settings.model_file,
		vocab_file=settings.vocab_file,
	)
	accent_settings = AccentSettings()
	app.state.f5_engine = F5Engine(f5_settings, accent_settings, settings.voices_dir)
	yield
	app.state.f5_engine = None


app = FastAPI(lifespan=lifespan)



@app.get("/voices")
async def get_voices(f5_engine: F5Engine = Depends(lambda: app.state.f5_engine)) -> Voices:
	return Voices(voices=[Voice(name=voice) for voice in f5_engine.voices])

@app.get("/models")
async def get_model(f5_engine: F5Engine = Depends(lambda: app.state.f5_engine)) -> Models:
	return Models(models=[Model(name=f5_engine.model_name)])

@app.post(
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
async def generate_audio_speech(request: AudioSpeechRequest, f5_engine: F5Engine = Depends(lambda: app.state.f5_engine)):
	if request.model != f5_engine.model_name:
		raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
	if request.voice not in f5_engine.voices:
		raise HTTPException(status_code=400, detail=f"Voice {request.voice} not found")
	if request.format not in ["pcm16", "wav"]:
		raise HTTPException(status_code=400, detail=f"Format {request.format} not supported")
	audio_bytes, sample_rate, duration = f5_engine.generate(voice=request.voice, gen_text=request.input, response_format=request.format)
	media_type = "audio/wav" if request.format == "wav" else "audio/pcm"
	headers = {
		"x-audio-sample-rate": str(sample_rate),
		"x-audio-duration-seconds": f"{duration:.6f}",
	}
	return Response(content=audio_bytes, media_type=media_type, headers=headers)

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
	input: str = Query(..., description="Input text"),
	voice: str = Query(..., description="Voice name"),
	model: str = Query(..., description="Model name"),
	format: Literal["pcm16", "wav"] = Query("wav", description="Format"),
	f5_engine: F5Engine = Depends(lambda: app.state.f5_engine),
):
	if model != f5_engine.model_name:
		raise HTTPException(status_code=400, detail=f"Model {model} not found")
	if voice not in f5_engine.voices:
		raise HTTPException(status_code=400, detail=f"Voice {voice} not found")
	if format not in ["pcm16", "wav"]:
		raise HTTPException(status_code=400, detail=f"Format {format} not supported")
	audio_bytes, sample_rate, duration = f5_engine.generate(voice=voice, gen_text=input, response_format=format)
	media_type = "audio/wav" if format == "wav" else "audio/pcm"
	headers = {
		"x-audio-sample-rate": str(sample_rate),
		"x-audio-duration-seconds": f"{duration:.6f}",
	}
	return Response(content=audio_bytes, media_type=media_type, headers=headers)

if __name__ == "__main__":
	import uvicorn
	settings = EnvSettings()
	uvicorn.run(app, host=settings.host, port=settings.port)






