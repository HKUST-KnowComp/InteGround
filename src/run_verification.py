import os
from data import get_datasets
from tqdm import tqdm
from baseline import ask
from utils import write_json
from config import verification_models
from random import sample, shuffle, seed


random_state = 20240627

result_dir = os.path.join("tmp", "verification")

out_fn = os.path.join(result_dir, "verification_results.json")

datasets = get_datasets()

os.makedirs(result_dir, exist_ok=True)


# make test set

print('making evaluation test set for model')
seed(random_state)

verification_test = {}
for key in datasets:
    verification_test[key] = []
    for data_id, item in enumerate(tqdm(datasets[key])):
        hyp = item['hypothesis']
        all_e = item['all_evidence']
        gt_ids = item['gt_ids']
        if isinstance(gt_ids[0], list): # multiple gt sets
            gt_ids = gt_ids[0]

        distractor_ids = [i for i in range(len(all_e)) if i not in gt_ids]

        type_1 = [all_e[id_] for id_ in gt_ids] # informative + succinct
        verification_test[key].append({
            "e1": type_1,
            "e2": hyp,
            "tag": "type_1",
            "dataset_id": data_id,
            "label": "entailment"
        })

        n_redundance = max(1, len(type_1)//2)
        redun_ids = sample(distractor_ids, n_redundance)
        type_2 = [all_e[id_] for id_ in gt_ids+redun_ids]   # informative with redundancy
        shuffle(type_2)
        verification_test[key].append({
            "e1": type_2,
            "e2": hyp,
            "tag": "type_2",
            "dataset_id": data_id,
            "label": "entailment"
        })

        n_incomplete = max(1, len(type_1)//2)
        incomplete_ids = sample(gt_ids, n_incomplete)
        type_3 = [all_e[id_] for id_ in incomplete_ids]     # incomplete information
        shuffle(type_3)
        verification_test[key].append({
            "e1": type_3,
            "e2": hyp,
            "tag": "type_3",
            "dataset_id": data_id,
            "label": "not entailment"
        })

        repeat = 0
        while repeat < 10:
            uninformative_ids = sample(range(len(all_e)), len(gt_ids))
            if not set(gt_ids).issubset(set(uninformative_ids)) and \
                not set(uninformative_ids).issubset(set(gt_ids)):
                    break
            repeat += 1 
        type_4 = [all_e[id_] for id_ in uninformative_ids]  # not informative
        verification_test[key].append({
            "e1": type_4,
            "e2": hyp,
            "tag": "type_4",
            "dataset_id": data_id,
            "label": "not entailment"
        })

# Running results
for model in verification_models:
    print("Running results for:", model)
    for key in verification_test:
        print(key)
        for item in tqdm(verification_test[key]):
            if model in item:   # skip tested
                continue
            
            retry = 0
            while retry < 3:
                try:
                    res = ask(item['e1'], item['e2'], model=model, mode='two-way')
                    item[model] = res
                    break
                except Exception as e:
                    print(e)
                retry += 1
                
            # break
            
        write_json(out_fn, verification_test)


