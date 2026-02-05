from typing import List
from pydantic import BaseModel, Field

# class SpeechRequest(BaseModel):
# 	model: Optional[str] = Field(None, description="Model name")
# 	input: str = Field(..., description="Text to synthesize")
# 	voice: Optional[str] = Field(None, description="Voice folder name under voices dir")
# 	format: Optional[str] = Field("wav", description="Output audio format")

class Voice(BaseModel):
	name: str = Field(..., description="Voice folder name under voices dir")

class Voices(BaseModel):
	voices: List[Voice] = Field(..., description="List of voices")



