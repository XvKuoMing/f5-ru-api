from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
	dim: int = 1024
	depth: int = 22
	heads: int = 16
	ff_mult: int = 2
	text_dim: int = 512
	conv_layers: int = 4


class AppSettings(BaseSettings):
	# Server
	api_host: str = Field(default="0.0.0.0", alias="API_HOST")
	api_port: int = Field(default=8000, alias="API_PORT")
	log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
		default="info", alias="LOG_LEVEL"
	)

	# Model
	model_repo: str = Field(default="ESpeech/ESpeech-TTS-1_RL-V2", alias="MODEL_REPO")
	model_file: str = Field(default="espeech_tts_rlv2.pt", alias="MODEL_FILE")
	vocab_file: str = Field(default="vocab.txt", alias="VOCAB_FILE")
	model_cfg_json: str = Field(
		default=json.dumps(
			{
				"dim": 1024,
				"depth": 22,
				"heads": 16,
				"ff_mult": 2,
				"text_dim": 512,
				"conv_layers": 4,
			}
		),
		alias="MODEL_CFG_JSON",
	)

	# Vocoder & device
	vocoder_name: Literal["vocos", "bigvgan"] = Field(default="vocos", alias="VOCODER_NAME")
	device_preference: Literal["auto", "cuda", "mps", "xpu", "cpu"] = Field(
		default="auto", alias="DEVICE_PREFERENCE"
	)

	# Voices
	voices_dir: str = Field(default="./voices", alias="VOICES_DIR")
	default_voice: str = Field(default="default", alias="DEFAULT_VOICE")

	# Generation defaults
	cross_fade_sec: float = Field(default=0.15, alias="CROSS_FADE_SEC")
	nfe_step: int = Field(default=32, alias="NFE_STEP")
	speed_default: float = Field(default=1.0, alias="SPEED_DEFAULT")
	target_sample_rate: int = Field(default=24000, alias="TARGET_SAMPLE_RATE")
	cfg_strength: float = Field(default=2.0, alias="CFG_STRENGTH")
	sway_sampling_coef: float = Field(default=-1.0, alias="SWAY_SAMPLING_COEF")
	stream_chunk_size: int = Field(default=2048, alias="STREAM_CHUNK_SIZE")

	# Optional tokens
	hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

	@property
	def model_cfg(self) -> ModelConfig:
		try:
			raw: Dict[str, Any] = json.loads(self.model_cfg_json)
			return ModelConfig(**raw)
		except (json.JSONDecodeError, ValidationError) as exc:
			raise RuntimeError(f"Invalid MODEL_CFG_JSON: {exc}") from exc

	@property
	def voices_path(self) -> Path:
		return Path(self.voices_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
	 return AppSettings()
