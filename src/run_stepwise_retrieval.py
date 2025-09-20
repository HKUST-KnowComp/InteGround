import os
from data import get_datasets
from tqdm import tqdm
from baseline import retrieve
from utils import write_json, load_json
from config import retriever_models, instruct_retrievers, retrieval_instructions

result_dir = os.path.join("tmp", "retrieval")
out_fn = os.path.join(result_dir, "stepwise_retrieval_results.json")

# Lazy testing: only add not yet tested models
if os.path.exists(out_fn):
    datasets = load_json(out_fn)
else:
    datasets = get_datasets()

os.makedirs(result_dir, exist_ok=True)

# Running results
for retriever in retriever_models:
    print("Running results for:", retriever)
    if retriever in instruct_retrievers:
        instructions = retrieval_instructions
    else:
        instructions = {"": None}
    for instr in instructions:
        for key in datasets:
            print(key)
            for item in tqdm(datasets[key]):
                ticker = retriever+instr
                if ticker in item:   # skip retrieved
                    continue

                hyp = item['hypothesis']
                all_e = item['all_evidence']

                gt_ids = item['gt_ids']
                if isinstance(gt_ids[0], list):
                    gt_ids = gt_ids[0]

                stepwise_result = {}
                gt_e = [all_e[id_] for id_ in gt_ids]
                for i in range(len(gt_ids)):
                    # add gt evidence to hypothesis step by step
                    stepwise_hyp = " ".join([hyp]+gt_e[:i])
                    res = retrieve(stepwise_hyp, all_e, topk=len(
                        all_e), model_name=retriever, task_description=instructions[instr])
                    res = {k: list(res[k]) for k in res}
                    stepwise_result[i] = {"gt_ids": gt_ids[i:], "result": {
                        "similarity": res['similarity'], 'indices': res['indices']}}
                item[ticker] = stepwise_result
            write_json(out_fn, datasets)
