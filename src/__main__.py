import numpy as np
from llm_sdk import Small_LLM_Model
import argparse
from typing import List, Dict, Any
from .schema import FunctionDefinition, Prompt, JSONStructure, GenerationState
import json
import os
from pathlib import Path


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call_me_maybe argument parser"
    )
    parser.add_argument('--input',
                        default="../data/input/function_calling_tests.json",
                        type=str,
                        help="It is the path to the json \
                        file containing function_calling_tests for input")
    parser.add_argument('--functions_definition',
                        default="../data/input/functions_definition.json",
                        type=str,
                        help="It is the path to the json file containing \
                        functions_definition used for function calling")
    parser.add_argument('--output',
                        default="../data/output/function_calling_results.json",
                        type=str,
                        help="It is the path to the json file containing \
                        the result of the program")
    return parser.parse_args()


def load_func_and_prompt(functions_definition, prompts) -> tuple[list[FunctionDefinition], list[Prompt]]:
    try:
        with open(functions_definition, "r", encoding="utf-8")\
                    as function_file:
            raw_json_into_function_dict = json.load(function_file)
            object_function_definition = [FunctionDefinition(**item) for
                                          item in raw_json_into_function_dict]
        with open(prompts, "r", encoding="utf-8") as prompts_file:
            raw_json_into_prompt_dict = json.load(prompts_file)
            object_prompts = [Prompt(**item) for item
                              in raw_json_into_prompt_dict]
    except Exception as e:
        raise RuntimeError("Error with file", e)
    return object_function_definition, object_prompts


def load_inverted_tokens(model: Small_LLM_Model) -> Dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as file:
            raw_json_vocab: Dict[int, str] = json.load(file)
    except Exception as e:
        raise RuntimeError("Error:", e)
    return {token_id: token_str for token_str,
            token_id in raw_json_vocab.items()}


def adding_constraint(
    current_str: str,
    logits: List[float],
    vocab_map: Dict[int, str],
    schema: JSONStructure,
    suffix_ids: List[int],
) -> np.ndarray:
    """Apply function-name and structural token constraints."""

    masked_logits = np.array(logits, dtype=np.float32)
    name_prefix = '"name": "'

    # ---------------------------------------------------------
    # Stage 1: Constrain the function name
    # ---------------------------------------------------------
    if (name_prefix in current_str and 
        schema.selected_function is None):
        partial_name = current_str.split(name_prefix)[-1]

        # Check whether a valid function name is complete.
        for name in schema.valid_function_names:
            if partial_name == name + '"':
                schema.selected_function = name
                schema.state = GenerationState.AFTER_FUNCTION_NAME
                break
        # Restrict the next token to valid name continuations.
        if schema.selected_function is None:
            valid_indices = []
            for token_id, token_str in vocab_map.items():
                candidate = partial_name + token_str
                if any((name + '"').startswith(candidate)
                    for name in schema.valid_function_names):
                    valid_indices.append(token_id)
            if valid_indices:
                mask = np.full_like(masked_logits, -float("inf"),)
                mask[valid_indices] = masked_logits[valid_indices]
                return mask
            raise ValueError(
                f"No valid continuation for function name: "
                f"{partial_name!r}"
            )

    # ---------------------------------------------------------
    # Stage 2: Force the fixed JSON suffix
    # ---------------------------------------------------------
    if schema.state == GenerationState.PARAMETERS_OBJECT:
        pass
    return masked_logits


def decode_constraint(prompt_obj: Prompt,
                      vocab_map: Dict[int, str],
                      functions: List[FunctionDefinition],
                      model: Small_LLM_Model) -> Dict[str, Any]:
    state_json = JSONStructure(functions)
    func_descriptions = "\n".join([f"- {f.name}: \
                        {f.description}" for f in functions])
    prefix_str = ("{\n"
                 f'  "prompt": {json.dumps(prompt_obj.prompt)},\n'
                 '  "name": "')
    system_prompt = (
        "Available Functions:\n"
        f"{func_descriptions}\n\n"
        "User Task: Translate the prompt into JSON.\n"
        f"User Prompt: {prompt_obj.prompt}\n\n"
        "JSON Output:\n"
        f"{prefix_str}"
    )
    suffix = (
                '",\n'
                '  "parameters": {\n'
                "  }\n"
                "}"
            )
    suffix_ids = model.encode(suffix)[0].tolist()
    input_id = model.encode(system_prompt)[0].tolist()
    generated_id: List[int] = []
    for _ in range(120):
        current_str = (
            prefix_str
            + (
                model.decode(generated_id)
                if generated_id
                else ""
            )
        )

        logits = model.get_logits_from_input_ids(
            input_id + generated_id
        )

        masked_logits = adding_constraint(
            current_str=current_str,
            logits=logits,
            vocab_map=vocab_map,
            schema=state_json,
            suffix_ids=suffix_ids,
        )

        if not np.isfinite(masked_logits).any():
            raise ValueError(
                "No valid token available for current state"
            )

        next_token_id = int(np.argmax(masked_logits))
        generated_id.append(next_token_id)

        full_decoded = (
            prefix_str + model.decode(generated_id)
        )

        print(
            f"Generated token: {next_token_id}, "
            f"Text: {repr(model.decode(generated_id))}"
        )

        # if state_json.suffix_position >= len(suffix_ids):
        #     break
    raw_json = full_decoded
    print("checking raw_json:",raw_json)
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Generated output is not valid JSON"
        ) from exc


if __name__ == "__main__":
    try:
        args = parse_arg()
        model = Small_LLM_Model()
        functions, prompts = load_func_and_prompt(args.functions_definition,
                                                  args.input)
        vocab_map = load_inverted_tokens(model)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for each_prompt in prompts:
            result_dict = decode_constraint(each_prompt, vocab_map,
                                            functions, model)
            results.append(result_dict)
            print(f"Generated Result for \
                '{each_prompt.prompt}':\n{result_dict}\n")
            break # testing: remove this later
        with open(args.output, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=2)
            print(f"Successfully saved output to {args.output}")
    except Exception as e:
        print("Error :", e)
