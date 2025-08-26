from pydantic import BaseModel, Field
from typing import List

class Model(BaseModel):
	name: str = Field(..., description="Model name")

class Models(BaseModel):
	models: List[Model] = Field(..., description="List of models")