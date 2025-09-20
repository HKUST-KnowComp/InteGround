import re
import torch
import json


def load_json(fn):
    with open(fn) as f:
        return json.loads(f.read())


def load_jsonl(fn):
    data = []
    with open(fn) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def write_json(fn, data):
    with open(fn, "w") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def get_llm_response(prompt, model="chatgpt", max_new_tokens=1024, stop_sequences=[]):
    """Prompting LLMs to get response.
    prompt <str>: the input text to be fed to the LLMs.
    model <str>: the model name.

    This method outputs the model generation (string).

    You should implement this part to proceed.
    """
    raise NotImplementedError(
        "Method 'get_llm_response' in 'utils.py' is not yet implemented!"
    )


def rank_eval(list_of_similarities, list_of_gt_ids, metrics=["recall@1"]):
    """ Compute ranking metrics. 
    list_of_similarities: a list of list; each sub-list is the similarity prediction of **ALL** instances.
    list_of_gt_ids: a list of list; each sub-list contains only the ground-truth ids.
    Example: 
        qrels_dict = {
            0: {"d_3": 1, "d_2": 1}
        }
        run_dict = {
            0: {"0": 0.9, "d_1": 0.8, "d_2": 0.7, "d_3": 0.95}
        }
        rank_eval([[0.9, 0.8, 0.7, 0.95]], [[2, 3]], [range(4)], ["recall@1", "recall@2", "recall@3", "recall@4", "ndcg", "mrr"])
        >>> {'recall@1': 0.5,
            'recall@2': 0.5,
            'recall@3': 0.5,
            'recall@4': 1.0,
            'ndcg': 0.8772153153380493,
            'mrr': 1.0}
    """
    from ranx import Qrels, Run
    from ranx import evaluate
    # assert all([len(sim) == len(all_) for sim, all_ in zip(list_of_similarities, list_of_all_ids)])
    # gts
    qrels_dict = {
        problem_id: {str(k): 1 for k in gt_ids}
        for problem_id, gt_ids in enumerate(list_of_gt_ids)
    }
    # predictions
    run_dict = {
        problem_id: {str(k): sims[k] for k in range(len(sims))}
        for problem_id, sims in enumerate(list_of_similarities)
    }
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    return evaluate(qrels, run, metrics)


def reindex(old_indice_group, mapping):
    # Reindex a list of indices with the mapping
    #
    # >>> gtids = [0, 1 , 2]
    # >>> similarities = [0.5, 0.4, 0.3, 0.2]
    # >>> indices = [0, 1, 2, 5]
    # >>> reindex([gtids, indices], {old:new for old, new in zip(indices, range(len(indices)))})
    # [[0, 1, 2], [0, 1, 2, 3]]
    return [[mapping[id_] for id_ in group] for group in old_indice_group]


def get_dict(text):
    return re.findall(r'{[\s\S]*?}', text)[0]


def get_list(text):
    return re.findall(r'\[[\s\S]*\]', text)[0]


def parse_json(text):
    if '```json' in text:
        json_text = re.findall(r'```json([\s\S]*)```', text)[0]
    else:
        json_text = text
    return eval(json_text)


def get_runstr_combinations(header='python3 finetune_lm.py', fixed_configs={}, comb_configs={}):
    """ Get all the combinations in `comb_configs`. 
    This is useful when trying to conduct multiple experiments with different params.

    Args:
        header <str>: The starting part of the command, e.g., python3 run.py 
        fixed_configs <dict: str->any>: The fixed part of parameters, you may also write these params in the header manually.
        comb_configs <dict: str-> list of any>: The combinitorial part of parameters. All the combinations within this dictionary will be considered.

    Returns:
        A list of all the combinations of runstrings.
    """
    from itertools import product

    all_runstrs = []

    # add fixed arguments
    string = header
    for key in fixed_configs:
        string += ' --'+key+' '+str(fixed_configs[key])

    # add all combinations
    key_list = list(comb_configs.keys())
    value_list = [comb_configs[key] for key in key_list]
    for vals in list(product(*value_list)):
        tmp_str = string
        for key, val in zip(key_list, vals):
            tmp_str += ' --'+key+' '+str(val)
        all_runstrs.append(tmp_str)

    return all_runstrs
