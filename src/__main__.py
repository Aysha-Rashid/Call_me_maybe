import numpy as np
from llm_sdk import Small_LLM_Model
import argparse
from typing import List, Dict, Any
from .schema import FunctionDefinition, Prompt, JSONStructure
import json


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
                        default="../data/output/function_calls.json",
                        type=str,
                        help="It is the path to the json file containing \
                        the result of the program")
    return parser.parse_args()


def load_func_and_prompt(functions_definition, prompts):
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


def adding_constraint(current_str: str, logits: List[float],
                      vocab_map: Dict[int, str],
                      schema: JSONStructure) -> np.ndarray:
    masked_logits = np.array(logits, dtype=np.float32)
    if len(current_str.strip()) == 0:
        for token_id, token_str in vocab_map.items():
            if not token_str.startswith("{") and token_str != "{":
                masked_logits[token_id] = -float('inf')
        return masked_logits
    if '"name": "' in current_str and schema.selected_function is None:
        valid_names = schema.valid_function_names
        # print("-----checking valid_names------", valid_names)
        for token_id, token_str in vocab_map.items():
            clean_token = token_str.replace('"', '').strip()
            if clean_token and not any(name.startswith(clean_token)
                                       for name in valid_names):
                masked_logits[token_id] = -float('inf')
            # else:
            #     print("correct clean_token:", clean_token)
    return masked_logits


def decode_constraint(prompt_obj: Prompt,
                      vocab_map: Dict[int, str],
                      functions: List[FunctionDefinition],
                      model: Small_LLM_Model) -> Dict[str, Any]:
    state_json = JSONStructure(functions)
    func_descriptions = "\n".join([f"- {f.name}: \
                        {f.description}" for f in functions])
    system_prompt = (
        f"Available Functions:\n{func_descriptions}\n\n"
        f"User Task: Translate the prompt into JSON.\n"
        f"User Prompt: {prompt_obj.prompt}\n"
        f"JSON Output:\n{{\n  \"prompt\": \
        \"{prompt_obj.prompt}\",\n  \"name\": \""
    )
    input_id = model.encode(system_prompt)[0].tolist()
    generated_id: List[int] = []
    prefix_str = f'{{\n  "prompt": "{prompt_obj.prompt}",\n  "name": "'
    for _ in range(120):
        current_str = prefix_str + (model.decode(generated_id)
                                    if generated_id else "")
        logits = model.get_logits_from_input_ids(input_id + generated_id)
        masked_logits = adding_constraint(current_str, logits,
                                          vocab_map, state_json)
        # print(masked_logits)
        next_token_id = int(np.argmax(masked_logits))
        generated_id.append(next_token_id)
        full_decoded = prefix_str + model.decode(generated_id)
        if full_decoded.endswith("}") and \
           full_decoded.count("{") == full_decoded.count("}"):
            break
    raw_json = prefix_str + full_decoded
    try:
        return json.loads(raw_json)
    except Exception as e:
        print("exception caught:", e)
        return {
            "prompt": prompt_obj.prompt,
            "name": state_json.selected_function or "None",
            "parameters": {}
        }


if __name__ == "__main__":
    try:
        args = parse_arg()
        model = Small_LLM_Model()
        functions, prompts = load_func_and_prompt(args.functions_definition,
                                                  args.input)
        vocab_map = load_inverted_tokens(model)
        results: List[Dict[str, Any]] = []
        for each_prompt in prompts:
            result_dict = decode_constraint(each_prompt, vocab_map,
                                            functions, model)
            results.append(result_dict)
            print(f"Generated Result for \
                '{each_prompt.prompt}':\n{result_dict}\n")
        with open(args.output, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=2)
            print(f"Successfully saved output to {args.output}")
    except Exception as e:
        print("Error :", e)
