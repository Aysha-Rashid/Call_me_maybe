import numpy as np
from llm_sdk import Small_LLM_Model
import argparse
from typing import List
from .schema import FunctionDefinition, Prompt
import json

def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    # 1. Initialize model and encode prompt to token IDs
    model = Small_LLM_Model()
    input_ids = model.encode(prompt)[0].tolist()
    
    generated_ids = []

    # 2. Auto-regressive generation loop
    for _ in range(max_new_tokens):
        # Get raw prediction scores (logits) for the current sequence
        logits = model.get_logits_from_input_ids(input_ids + generated_ids)
        
        # Pick the token ID with the highest score (greedy selection / argmax)
        next_token_id = int(np.argmax(logits))
        # Append to our generated sequence[cite: 2]
        generated_ids.append(next_token_id)
        # print("checking:", model.decode(generated_ids))


    # 3. Decode the generated token IDs back to a human-readable string[cite: 1, 2]
    full_output = model.decode(generated_ids)
    return full_output


def parse_arg() -> argparse.Namespace :
    parser = argparse.ArgumentParser(
        description="Call_me_maybe argument parser"
    )
    parser.add_argument('--input',
        default="../data/input/function_calling_tests.json",
        type=str,
        help="It is the path to the json file containing function_calling_tests for input"
    )
    parser.add_argument('--functions_definition',
        default="../data/input/functions_definition.json",
        type=str,
        help="It is the path to the json file containing functions_definition used for function calling"
    )
    parser.add_argument('--output',
        default="../data/output/function_calls.json",
        type=str,
        help="It is the path to the json file containing the result of the program" 
    )
    return parser.parse_args()


def load_functions_and_prompt(functions_definition, prompts):
    
    try:
        with open(functions_definition, "r", encoding="utf-8") as function_file:
            raw_json_into_function_dict = json.load(function_file)
            # print("printing raw_json_into_dict:",raw_json_into_dict)
            object_function_definition = [FunctionDefinition(**item) for item in raw_json_into_function_dict]
            print("object function defination:", object_function_definition)
        with open(prompts, "r", encoding="utf-8") as prompts_file:
            raw_json_into_prompt_dict = json.load(prompts_file)
            object_prompts = [Prompt(**item) for item in raw_json_into_prompt_dict]
    except Exception as e:
        raise RuntimeError("Error with file", e)
    return object_function_definition, object_prompts

if __name__ == "__main__":
    try:
        args = parse_arg()

        functions = load_functions_and_prompt(args.functions_definition, args.input)
        test_prompt =  \
            f'''
                User query: what is the sum of 12 and 13?
            '''
        # print(f"--- Prompt ---\n{test_prompt}\n")
        print(f"Loading functions from: {args.functions_definition}")
        print(f"Loading test queries from: {args.input}")
        print(f"Writing output to: {args.output}")
        # result = generate_text(test_prompt, max_new_tokens=30)
        # print(f"--- Model Output ---\n{result}")
    except Exception as e:
        print("Error :", e)
    # finally:
        #delete any memory allocation