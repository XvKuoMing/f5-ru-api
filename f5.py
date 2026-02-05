import os
import io
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
import asyncio
from contextlib import asynccontextmanager
from ruaccent import RUAccent
from f5_tts.infer.utils_infer import (
    infer_process,
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    chunk_text,
)
from f5_tts.model import DiT
from typing import Optional, Any, Literal, Iterator, Tuple
# from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class F5Settings:
    model_repo: str
    model_file: str
    vocab_file: str
    model_cfg: Optional[dict] = field(default_factory=lambda: dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4))

@dataclass
class AccentSettings:
    omograph_model_size: str = "turbo3.1"
    use_dictionary: bool = True
    tiny_mode: bool = False

@dataclass
class F5GenerationSettings:
    cross_fade_duration: float = 0.15
    nfe_step: int = 32
    speed: float = 1.0

@dataclass
class F5Voice:
    ref_audio: Any
    ref_text: str

class F5VoiceManager:

    def __init__(self, voices_dir: str, accent_settings: AccentSettings):
        self.voices_dir = voices_dir
        self.accentizer = RUAccent()
        logger.info(f"Loading accentizer")
        self.accentizer.load(
            omograph_model_size=accent_settings.omograph_model_size, 
            use_dictionary=accent_settings.use_dictionary, 
            tiny_mode=accent_settings.tiny_mode
        )
        logger.info(f"Accentizer loaded")
        self.voices = {}
        self.load_voices()
        
    def maybe_accentize(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if '+' in text:
            return text
        else:
            return self.accentizer.process_all(text)
    
    
    def load_voices(self):
        """
        expect that each each voice is a subdir in voices_dir, each voice has a ref.wav and ref.txt
        """
        for voice_dir in os.listdir(self.voices_dir):
            if not os.path.isdir(os.path.join(self.voices_dir, voice_dir)):
                continue
            ref_audio_path = os.path.join(self.voices_dir, voice_dir, "ref.wav")
            ref_text_path = os.path.join(self.voices_dir, voice_dir, "ref.txt")
            with open(ref_text_path, "r") as f:
                ref_text = f.read()
            ref_text = self.maybe_accentize(ref_text.strip())
            ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                ref_audio_path,
                ref_text
            )
            self.voices[voice_dir] = F5Voice(ref_audio_proc, processed_ref_text_final)

    
    def get_voice(self, voice_name: str) -> F5Voice:
        return self.voices[voice_name]



class F5Engine:

    @staticmethod
    def convert_to_pcm16(wave: torch.Tensor, sample_rate: int, format: Literal["pcm", "wav"] = "pcm") -> bytes:
        # Accept torch.Tensor or numpy.ndarray, normalize to float32 in [-1, 1]
        if isinstance(wave, torch.Tensor):
            wave_np = wave.detach().cpu().float().numpy()
        elif isinstance(wave, np.ndarray):
            wave_np = wave.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported wave type: {type(wave)}")

        # Shape normalization: prefer (num_samples, num_channels)
        if wave_np.ndim == 1:
            samples_by_channels = wave_np[:, None]  # (T, 1)
        elif wave_np.ndim == 2:
            # Common model output is (channels, samples); convert to (samples, channels)
            samples_by_channels = wave_np.T if wave_np.shape[0] <= 8 and wave_np.shape[0] < wave_np.shape[1] else wave_np
        else:
            raise ValueError(f"Invalid wave shape: {wave_np.shape}")

        # Clip to [-1, 1]
        samples_by_channels = np.clip(samples_by_channels, -1.0, 1.0)

        if format == "pcm":
            int16_samples = (samples_by_channels * 32767.0).astype(np.int16, copy=False)
            # Interleave channels for raw PCM
            return int16_samples.tobytes()
        elif format == "wav":
            buffer = io.BytesIO()
            sf.write(buffer, samples_by_channels, samplerate=sample_rate, subtype="PCM_16", format="WAV")
            return buffer.getvalue()
        else:
            raise ValueError(f"Invalid format: {format}")

    def __init__(self, settings: F5Settings, accent_settings: AccentSettings, voices_dir: str):
        self.settings = settings
        self.model_path = hf_hub_download(repo_id=settings.model_repo, filename=settings.model_file)
        self.vocab_path = hf_hub_download(repo_id=settings.model_repo, filename=settings.vocab_file)
        
        logger.info(f"Loading model from {self.model_path} and {self.vocab_path}")
        self.model = load_model(DiT, settings.model_cfg, self.model_path, vocab_file=self.vocab_path)
        logger.info(f"Model loaded")
        logger.info(f"Loading vocoder")
        self.vocoder = load_vocoder()
        logger.info(f"Vocoder loaded")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.vocoder.to(self.device)
        
        logger.info(f"Loading voice manager")
        self.voice_manager = F5VoiceManager(voices_dir, accent_settings)
        logger.info(f"Voice manager loaded")

        self.gen_config = F5GenerationSettings()
    
    @property
    def model_name(self) -> str:
        return self.settings.model_repo
    
    @property
    def voices(self) -> list[str]:
        return list(self.voice_manager.voices.keys())
    
    def generate(self, 
        voice: str,
        gen_text: str, 
        response_format: str = "pcm",
        gen_config: Optional[F5GenerationSettings] = None,
    ) -> bytes:
        """
        assuming ref audio and text are already processed
        gen_text is the text to generate
        gen_config is the generation config
        returns the generated audio
        """
        gen_config = gen_config or self.gen_config
        voice_obj = self.voice_manager.get_voice(voice)
        ref_audio = voice_obj.ref_audio
        ref_text = voice_obj.ref_text
        gen_text = self.voice_manager.maybe_accentize(gen_text)
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text.strip(),
            self.model,
            self.vocoder,
            cross_fade_duration=gen_config.cross_fade_duration,
            nfe_step=gen_config.nfe_step,
            speed=gen_config.speed,
        )
        # Compute duration robustly across shapes/types
        try:
            if isinstance(final_wave, torch.Tensor):
                num_samples = final_wave.shape[-1]
            elif isinstance(final_wave, np.ndarray):
                num_samples = final_wave.shape[0] if final_wave.ndim == 1 else max(final_wave.shape)
            else:
                num_samples = None
        except Exception:
            num_samples = None

        duration = (float(num_samples) / float(final_sample_rate)) if num_samples is not None else 0.0
        return self.convert_to_pcm16(final_wave, final_sample_rate, response_format), final_sample_rate, duration

    def generate_stream(
        self,
        voice: str,
        gen_text: str,
        response_format: Literal["pcm"] = "pcm",
        gen_config: Optional[F5GenerationSettings] = None,
        chunk_size: int = 2048,
    ) -> Tuple[Iterator[bytes], int]:
        """
        Stream audio as PCM16 chunks.

        Returns (generator, sample_rate).
        """
        if response_format != "pcm":
            raise ValueError("Streaming currently supports only 'pcm'.")

        gen_config = gen_config or self.gen_config
        voice_obj = self.voice_manager.get_voice(voice)
        ref_audio_path = voice_obj.ref_audio
        ref_text = voice_obj.ref_text
        gen_text_accented = self.voice_manager.maybe_accentize(gen_text)

        # Prepare batching similarly to infer_process
        audio_tensor, sr = torchaudio.load(ref_audio_path)
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        # Estimate max chars per chunk using the same heuristic
        max_chars = int(len(ref_text.encode("utf-8")) / (audio_tensor.shape[-1] / sr) * (22 - audio_tensor.shape[-1] / sr) * gen_config.speed)
        gen_text_batches = chunk_text(gen_text_accented.strip(), max_chars=max_chars)

        def pcm_generator() -> Iterator[bytes]:
            for chunk, sample_rate in infer_batch_process(
                (audio_tensor, sr),
                ref_text,
                gen_text_batches,
                self.model,
                self.vocoder,
                progress=None,
                cross_fade_duration=gen_config.cross_fade_duration,
                nfe_step=gen_config.nfe_step,
                speed=gen_config.speed,
                device=str(self.device),
                streaming=True,
                chunk_size=chunk_size,
            ):
                yield self.convert_to_pcm16(chunk, sample_rate, format="pcm")

        # The infer path always resamples to 24000 inside utils, but we return the yielded rate
        # from the generator via header at the API level using the known target rate 24000.
        return pcm_generator(), 24000
        
        
        
        
        
        
        

class F5EnginePool:

    def __init__(self, settings: F5Settings, accent_settings: AccentSettings, voices_dir: str, num_instances: int = 1):
        if num_instances < 1:
            raise ValueError("num_instances must be >= 1")
        self._instances: list[F5Engine] = [F5Engine(settings, accent_settings, voices_dir) for _ in range(num_instances)]
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(len(self._instances)):
            self._available.put_nowait(i)

    @property
    def model_name(self) -> str:
        return self._instances[0].model_name

    @property
    def voices(self) -> list[str]:
        return self._instances[0].voices

    async def _acquire_index(self) -> int:
        idx = await self._available.get()
        return idx

    def _release_index(self, idx: int) -> None:
        self._available.put_nowait(idx)

    @asynccontextmanager
    async def acquire(self):
        idx = await self._acquire_index()
        try:
            yield self._instances[idx]
        finally:
            self._release_index(idx)

    async def generate(
        self,
        *,
        voice: str,
        gen_text: str,
        response_format: str = "pcm",
        gen_config: Optional[F5GenerationSettings] = None,
    ) -> tuple[bytes, int, float]:
        async with self.acquire() as engine:
            return await asyncio.to_thread(
                engine.generate,
                voice,
                gen_text,
                response_format,
                gen_config,
            )

    async def start_stream(
        self,
        *,
        voice: str,
        gen_text: str,
        response_format: Literal["pcm"] = "pcm",
        gen_config: Optional[F5GenerationSettings] = None,
        chunk_size: int = 2048,
    ) -> tuple[Iterator[bytes], int]:
        idx = await self._acquire_index()
        engine = self._instances[idx]
        try:
            base_gen, sample_rate = engine.generate_stream(
                voice=voice,
                gen_text=gen_text,
                response_format=response_format,
                gen_config=gen_config,
                chunk_size=chunk_size,
            )

            def wrapped_gen() -> Iterator[bytes]:
                try:
                    for chunk in base_gen:
                        yield chunk
                finally:
                    # Ensure engine slot is released when streaming ends
                    self._release_index(idx)

            return wrapped_gen(), sample_rate
        except Exception:
            # In case of early failure, release the slot
            self._release_index(idx)
            raise