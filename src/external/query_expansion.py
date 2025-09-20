""" The zero-shot prompt is adapted from HyDE's official repo: https://github.com/texttron/hyde/blob/main/src/hyde/promptor.py
"""

import sys
sys.path.append("..")

from utils import get_dict, get_llm_response
from nltk import sent_tokenize

zero_shot_prompt = """Please write a passage to support/refute the claim.
Claim: {}
Passage:"""

few_shot_prompt = """Please write a passage to support/refute the claim.
Claim: {}
Passage (in the format "{{{{"passage": "PASSAGE TEXT"}}}}"):  
{{{{"passage": "{}"}}}}

Claim: {}
Passage (in the format "{{{{"passage": "PASSAGE TEXT"}}}}"): 
{{{{"passage": "{}"}}}}
"""

few_shot_postfix = """
Claim: {}
Passage (in the format "{{"passage": "PASSAGE TEXT"}}"): """

def get_query_expansion_few_shot_prompt(train_datasets, dataset="entailmentbank"):
    """ Return a few-shot prompt template for a given dataset. """
    from config import max_evidence_num, min_gt_num
    from data import get_datasets
    train_datasets = get_datasets("train", max_evidence_num, min_gt_num=min_gt_num)

    instances = train_datasets[dataset][:2]
    claims = [item['hypothesis'] for item in instances]
    passages = [item['gt_evidence'] for item in instances]
    passages = ["\\n ".join(item if isinstance(item[0], str) else item[0]) for item in passages]
    prompt = few_shot_prompt.format(claims[0], passages[0], claims[1], passages[1]) + few_shot_postfix
    return prompt

def query_expansion(input_text, model="claude-v3-sonnet", mode="zeroshot"):
    """ Prompt LLMs for zero-shot / few-shot query expansion. """

    if mode == "zeroshot":
        prompt = zero_shot_prompt.format(input_text)
    elif mode.startswith("fewshot"):
        raise NotImplementedError("Fewshot query expansion not yet implemented.")
    else:
        raise ValueError("Wrong mode specified: {}".format(mode))
    
    retry = 0
    while retry < 3:
        try:
            res = get_llm_response(prompt, model=model)

            if "\n\n" in res:       # Remove prefixes like "here is a passage supporting the claim \n\n:"
                res = res[res.index("\n\n")+2:]
            # res = eval(get_dict(res))['passage']
            break
        except Exception as e:
            print("[In query expansion]", e, retry)
            retry += 1
            res = ""
    return sent_tokenize(res)