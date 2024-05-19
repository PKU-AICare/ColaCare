import os
import json
import re

import openai
import tiktoken

from config import tech_config, deep_config
from template import *
from retriever import Retriever

config = deep_config
if openai.api_key is None:
    openai.api_type = config["api_type"]
    openai.api_base = config["api_base"] 
    openai.api_key = config["api_key"]


class Generator:
    def __init__(self, model_name: str=None, retmax=100):
        if model_name is None:
            self.model = config['model']
        else:
            self.model = model_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_length = 32768
        self.context_length = 30000
        self.retmax = retmax
    
    def generate_keywords(self, question):
        messages=[
            {"role": "system", "content": keywords_system},
            {"role": "user", "content": keywords_user.render(question=question)}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            raise Exception(f"Failed to generate keywords: {e}")
        ans = response["choices"][0]["message"]["content"]
        ans = re.sub("\s+", " ", ans)
        try:
            ans = json.loads(ans)
        except:
            ans = ans.replace("```", "").replace("json", "")
            ans = json.loads(ans)
        return ans
    
    def generate_answer(self, hcontext):
        keywords = self.generate_keywords(hcontext)
        keywords = ', '.join(keywords['input'])
        retriver = Retriever(keywords, retmax=self.retmax)
        retrieved_contexts, scores = retriver.get_relevant_documents(hcontext)
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_contexts[idx]["title"], retrieved_contexts[idx]["abstract"]) for idx in range(len(retrieved_contexts))]
        context = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
        
        messages=[
            {"role": "system", "content": ensemble_system},
            {"role": "user", "content": ensemble_user.render(context=context, hcontext=hcontext)}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )
        except Exception as e:
            raise Exception(f"Failed to generate answer: {e}")
        ans = response["choices"][0]["message"]["content"]
        ans = re.sub("\s+", " ", ans)
        try:
            ans = json.loads(ans)
        except:
            ans = ans.replace("```", "").replace("json", "")
            ans = json.loads(ans)
        return ans, keywords, retrieved_contexts, scores, messages