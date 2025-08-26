from pydantic import BaseModel, Field
from typing import Literal

class AudioSpeechRequest(BaseModel):
    input: str = Field(..., description="Input text")
    voice: str = Field(..., description="Voice name")
    model: str = Field(..., description="Model name")
    format: Literal["pcm16", "wav"] = Field(..., description="Format")

class AudioSpeechResponse(BaseModel):
    audio: str = Field(..., description="Base64-encoded generated audio")
    sample_rate: int = Field(..., description="Sample rate")
    duration: float = Field(..., description="Duration in seconds")