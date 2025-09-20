import os
from data import get_datasets
from tqdm import tqdm
from baseline import retrieve
from utils import write_json, load_json
from config import retriever_models, instruct_retrievers, retrieval_instructions

result_dir = os.path.join("tmp", "retrieval")
out_fn = os.path.join(result_dir, "retrieval_results.json")

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

                res = retrieve(hyp, all_e, topk=len(
                    all_e), model_name=retriever, task_description=instructions[instr])
                # item[ticker] = list(res['similarity'])
                res = {k: list(res[k]) for k in res}
                item[ticker] = {"similarity": res['similarity'],
                                'indices': res['indices']}
            write_json(out_fn, datasets)
