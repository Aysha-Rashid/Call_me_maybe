from pydantic import BaseModel
from typing import Dict, Any

class ParameterSchema(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: Dict[str, Any]

class Prompt(BaseModel):
    prompt: str

class Result(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
    