from pydantic import BaseModel
from typing import Dict, List, Set, Any


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


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


class JSONStructure:
    def __init__(self, functions: List[FunctionDefinition]):
        self.functions = functions
        self.valid_function_names: Set[str] = {f.name for f in functions}
        self.param_keys_by_func: Dict[str, Set[str]] = {
            f.name: set(f.parameters.keys()) for f in functions
        }
        self.selected_function: str | None = None
