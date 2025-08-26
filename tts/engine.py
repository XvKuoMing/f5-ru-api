from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from f5_tts.infer.utils_infer import (
	 infer_batch_process,
	 load_model,
	 load_vocoder,
	 preprocess_ref_audio_text,
)
from f5_tts.model import DiT

from .config import AppSettings, get_settings


@dataclass
class VoiceRef:
	name: str
	audio_path: Path
	text_path: Path


class TTSEngine:
	def __init__(self, settings: Optional[AppSettings] = None) -> None:
		self.settings = settings or get_settings()
		self.device = self._resolve_device()
		self._model = None
		self._vocoder = None
		self._voices: Dict[str, VoiceRef] = {}

	def _resolve_device(self) -> str:
		if self.settings.device_preference != "auto":
			return self.settings.device_preference
		if torch.cuda.is_available():
			return "cuda"
		if hasattr(torch, "xpu") and torch.xpu.is_available():
			return "xpu"
		if torch.backends.mps.is_available():
			return "mps"
		return "cpu"

	def load(self) -> None:
		if self._model is None:
			model_path, vocab_path = self._ensure_model_files()
			self._model = load_model(
				DiT,
				self.settings.model_cfg.model_dump(),
				model_path,
				vocab_file=str(vocab_path),
			)
		if self._vocoder is None:
			self._vocoder = load_vocoder(self.settings.vocoder_name, device=self.device)
		self._scan_voices()

	def _ensure_model_files(self) -> Tuple[str, str]:
		from huggingface_hub import hf_hub_download, snapshot_download

		model_path: Optional[str] = None
		vocab_path: Optional[str] = None

		try:
			model_path = hf_hub_download(repo_id=self.settings.model_repo, filename=self.settings.model_file, token=self.settings.hf_token)
			vocab_path = hf_hub_download(repo_id=self.settings.model_repo, filename=self.settings.vocab_file, token=self.settings.hf_token)
		except Exception:
			local_dir = f"cache_{self.settings.model_repo.replace('/', '_')}"
			snapshot_dir = snapshot_download(repo_id=self.settings.model_repo, local_dir=local_dir, token=self.settings.hf_token)
			mp = Path(snapshot_dir) / self.settings.model_file
			vp = Path(snapshot_dir) / self.settings.vocab_file
			if mp.exists():
				model_path = str(mp)
			if vp.exists():
				vocab_path = str(vp)
		if not model_path or not Path(model_path).exists():
			raise FileNotFoundError(f"Model not found: {model_path}")
		if not vocab_path or not Path(vocab_path).exists():
			raise FileNotFoundError(f"Vocab not found: {vocab_path}")
		return model_path, vocab_path

	def _scan_voices(self) -> None:
		self._voices.clear()
		voices_root = self.settings.voices_path
		voices_root.mkdir(parents=True, exist_ok=True)
		for entry in voices_root.iterdir():
			if not entry.is_dir():
				continue
			audio: Optional[Path] = None
			text: Optional[Path] = None
			for cand in entry.iterdir():
				if cand.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"} and audio is None:
					audio = cand
				if cand.suffix.lower() in {".txt"} and text is None:
					text = cand
			if audio and text:
				self._voices[entry.name] = VoiceRef(name=entry.name, audio_path=audio, text_path=text)

	@property
	def voices(self) -> List[str]:
		return sorted(self._voices.keys())

	def get_voice(self, voice_name: Optional[str]) -> VoiceRef:
		if not voice_name:
			voice_name = self.settings.default_voice
		if voice_name not in self._voices:
			raise KeyError(f"Voice '{voice_name}' not found. Available: {', '.join(self.voices)}")
		return self._voices[voice_name]

	def synthesize_numpy(
		self,
		text: str,
		voice_name: Optional[str] = None,
		*,
		cross_fade_sec: Optional[float] = None,
		nfe_step: Optional[int] = None,
		speed: Optional[float] = None,
	) -> Tuple[np.ndarray, int]:
		if self._model is None or self._vocoder is None:
			self.load()
		voice = self.get_voice(voice_name)
		with open(voice.text_path, "r", encoding="utf-8") as f:
			ref_text = f.read().strip()
		ref_audio_preproc_path, processed_ref_text = preprocess_ref_audio_text(str(voice.audio_path), ref_text)
		ref_audio_tensor, ref_sr = torchaudio.load(ref_audio_preproc_path)
		for result in infer_batch_process(
			ref_audio=(ref_audio_tensor, ref_sr),
			ref_text=processed_ref_text,
			gen_text_batches=[text],
			model_obj=self._model,
			vocoder=self._vocoder,
			mel_spec_type="vocos",
			progress=None,
			target_rms=0.1,
			cross_fade_duration=cross_fade_sec if cross_fade_sec is not None else self.settings.cross_fade_sec,
			nfe_step=nfe_step if nfe_step is not None else self.settings.nfe_step,
			cfg_strength=self.settings.cfg_strength,
			sway_sampling_coef=self.settings.sway_sampling_coef,
			speed=speed if speed is not None else self.settings.speed_default,
			fix_duration=None,
			device=self.device,
			streaming=False,
		):
			if isinstance(result, tuple) and len(result) == 3:
				final_wave, final_sr, _ = result
				return final_wave.astype(np.float32), final_sr
		return np.zeros((0,), dtype=np.float32), self.settings.target_sample_rate

	def synthesize_stream_pcm16(
		self,
		text: str,
		voice_name: Optional[str] = None,
		*,
		chunk_size: Optional[int] = None,
		nfe_step: Optional[int] = None,
		speed: Optional[float] = None,
	) -> Generator[bytes, None, None]:
		if self._model is None or self._vocoder is None:
			self.load()
		voice = self.get_voice(voice_name)
		with open(voice.text_path, "r", encoding="utf-8") as f:
			ref_text = f.read().strip()
		ref_audio_preproc_path, processed_ref_text = preprocess_ref_audio_text(str(voice.audio_path), ref_text)
		ref_audio_tensor, ref_sr = torchaudio.load(ref_audio_preproc_path)

		for chunk, sample_rate in infer_batch_process(
			ref_audio=(ref_audio_tensor, ref_sr),
			ref_text=processed_ref_text,
			gen_text_batches=[text],
			model_obj=self._model,
			vocoder=self._vocoder,
			mel_spec_type="vocos",
			progress=None,
			target_rms=0.1,
			cross_fade_duration=self.settings.cross_fade_sec,
			nfe_step=nfe_step if nfe_step is not None else self.settings.nfe_step,
			cfg_strength=self.settings.cfg_strength,
			sway_sampling_coef=self.settings.sway_sampling_coef,
			speed=speed if speed is not None else self.settings.speed_default,
			fix_duration=None,
			device=self.device,
			streaming=True,
			chunk_size=chunk_size if chunk_size is not None else self.settings.stream_chunk_size,
		):
			pcm16 = np.clip(chunk, -1.0, 1.0)
			pcm16 = (pcm16 * 32767.0).astype(np.int16)
			yield pcm16.tobytes()

	def encode_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
		buf = io.BytesIO()
		sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
		return buf.getvalue()


_engine_singleton: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
	global _engine_singleton
	if _engine_singleton is None:
		_engine_singleton = TTSEngine()
		_engine_singleton.load()
	return _engine_singleton
