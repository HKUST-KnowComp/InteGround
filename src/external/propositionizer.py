""" Adapted from https://github.com/chentong0/factoid-wiki """

import torch
import json
from transformers import pipeline

import sys
sys.path.append("..")
from utils import get_dict
from mykits import get_llm_response

# The following template is adapted from the one used in this paper: https://arxiv.org/pdf/2311.04335
proposition_template = \
"""Given the following sentence, tell me what claims they are making. Please split the sentence as much as possible, but do not include information not in the sentence.

Sentence: The Andy Warhol Museum in his hometown, Pittsburgh, Pennsylvania, contains an extensive permanent collection of art.
{{"Claims": ["The Andy Warhol Museum is in Pittsburgh.", "Andy Warhol’s hometown is in Pittsburgh.", "Pittsburgh is in Pennsylvania.", "The Andy Warhol Museum contains an extensive permanent collection of art."]}}

Sentence: {}"""

def init_propositionizer(model="chentong00/propositionizer-wiki-flan-t5-large", device='cuda:0', max_new_tokens=512):
    global propositionizer
    
    propositionizer = pipeline("text2text-generation", model=model, max_new_tokens=max_new_tokens, device=device)
    
def proposition_decompose(input_text, model="chentong00/propositionizer-wiki-flan-t5-large", device='cuda:0', max_new_tokens=512):
    if model == "chentong00/propositionizer-wiki-flan-t5-large":
        global propositionizer

        if 'propositionizer' not in globals():
            print("Initing model:", model)
            init_propositionizer(model=model, device=device)
            
        elif globals()['propositionizer'].model.name_or_path != model:
            del propositionizer
            torch.cuda.empty_cache()
            print("Initing model:", model)
            init_propositionizer(model=model, device=device, max_new_tokens=max_new_tokens)

        title = ""
        section = ""
        input_text = f"Title: {title}. Section: {section}. Content: {input_text}"
        propositions = json.loads(propositionizer(input_text)[0]['generated_text'])

    else:
        # LLM based
        retry = 0
        propositions = []
        while retry < 3:
            try:
                res = get_llm_response(proposition_template.format(input_text), model=model)
                propositions = eval(get_dict(res))['Claims']
                break
            except Exception as e:
                print("[In proposition decomposition]", e, retry)
                retry += 1

    return propositions