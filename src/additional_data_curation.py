import os
import json
from data import get_datasets
from tqdm import tqdm
from utils import get_dict, write_json, get_llm_response

split_name = "test"
max_evidence_num = 200
min_gt_num = 2

datasets = get_datasets(split_name, max_evidence_num, min_gt_num=min_gt_num)


modify_prompt = \
    """Modify the statement to suggest otherwise that partially contradicts the original:

Statement: A pound sterling is fiat money.
{{"Modified statement": "A pound sterling is a kind of cryptocurrency."}}

Statement: Dogs have sensitive ears that can hear as far as a quarter of a mile away.
{{"Modified statement": "Dogs have average hearing abilities and cannot hear beyond a few yards."}}

Statement: Relay races are athletic track and field events.
{{"Modified statement": "Relay races are intellectual board games."}}

Statement: {}"""


def modify_statement(statement, model='llama3-70b-instruct'):
    dict_str = get_dict(get_llm_response(
        modify_prompt.format(statement), model=model))
    return json.loads(dict_str)['Modified statement']

augmented_data = "/home/ubuntu/EventGrounding/new_situation_ground/tmp/augmented_data"
for key in datasets:

    save_fn = os.path.join(augmented_data, f"{key}.json")
    print("Running", save_fn)

    tmp = []
    for item in tqdm(datasets[key]):
        retry = 0
        while retry < 3:
            modified = ""
            try:
                hyp = item['hypothesis']
                modified = modify_statement(hyp)
                break

            except Exception as e:
                retry += 1
                print(e, "retry=", retry)

        tmp.append(modified)

    write_json(save_fn, tmp)
