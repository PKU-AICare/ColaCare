import sys
sys.path.append('.')
import re
import json

import openai
import ollama

from template import *
from models.ckd_utils import generate_prompt
from config import tech_config, deep_config

config = deep_config
if openai.api_key is None:
    openai.api_base = config["api_base"]
    openai.api_key = config["api_key"]


def extract_and_parse_json(text):
    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    
    if match:
        json_string = match.group(1).strip()
        return json.loads(json_string)
    
    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        return json.loads(json_string)

    raise ValueError("No valid JSON content found.")


def generate_keywords(model, context):
    messages=[
        {"role": "system", "content": keywords_system},
        {"role": "user", "content": keywords_user.render(context=context)}
    ]
    if "llama" in model.lower():
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        ans = response["message"]["content"]
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
        ans = response["choices"][0]["message"]["content"]
    
    try:
        ans = extract_and_parse_json(ans)
        return ans
    except Exception as e:
        print(ans)
        raise e