import os
from tqdm import tqdm
from data import get_datasets
from utils import write_json, load_json
from baseline import get_planner, retrieve

split_name = "test"

result_dir = os.path.join("tmp", "planning_history_agnostic")
os.makedirs(result_dir, exist_ok=True)

# planners = [
#     ("entailer", "allenai/entailer-large"),
#     ("propositionizer", None),
#     ("factscore", "claude-v3.5-sonnet"),
#     ("factscore", "llama3.1-70b-instruct"),
#     ("query_expansion", "claude-v3.5-sonnet"),
#     ("query_expansion", "llama3.1-70b-instruct")
# ]

planners = [
    ("entailer", "claude-v3.5-sonnet"),
    ("entailer", "llama3.1-70b-instruct"),
    ("propositionizer", "claude-v3.5-sonnet"),
    ("propositionizer", "llama3.1-70b-instruct"),
    ("factscore", "claude-v3.5-sonnet"),
    ("factscore", "llama3.1-70b-instruct"),
    ("query_expansion", "claude-v3.5-sonnet"),
    ("query_expansion", "llama3.1-70b-instruct")
]

plan_fn = os.path.join(result_dir, "plans.json")

retrievers = [
    "BM25", 
    "sentence-transformers/sentence-t5-large", 
    "all-MiniLM-L6-v2", 
    "intfloat/multilingual-e5-large-instruct"
]

do_plan = True
# do_plan = False
do_retrieve = True
# do_retrieve = False


if do_plan:
    # Generate all plans

    if os.path.exists(plan_fn):
        datasets = load_json(plan_fn)
    else:
        datasets = get_datasets()

    print("Generating planning queries.")
    for planner_type, planner_model_name in planners:
        planner_key = planner_type + "|" + (planner_model_name if planner_model_name else "")
        print(planner_key)

        planner = get_planner(planner_type, planner_model_name)

        for data_split in datasets:
            for i, item in enumerate(tqdm(datasets[data_split])):
                if "plans" not in item:
                    item['plans'] = {}

                if planner_key in item['plans']:
                    continue
                else:
                    plans = planner(item['hypothesis'])
                    item['plans'][planner_key] = plans

                if i % 20 == 0:
                    write_json(plan_fn, datasets)
            write_json(plan_fn, datasets)


if do_retrieve:
    # Retrieve according to planned queries

    assert os.path.exists(plan_fn)
    datasets = load_json(plan_fn)

    def retrieval_wrapper_func(query, corpus, topk, model_name):
        res = retrieve(query, corpus, topk, model_name)
        res = {k: list(res[k]) for k in res}
        return {"similarity": res['similarity'], 'indices': res['indices']}

    for retriever in retrievers:
        for data_split in datasets:
            for i, item in enumerate(tqdm(datasets[data_split])):
                if "plans" in item:
                    if "retrieval_with_plans" not in item:
                        item["retrieval_with_plans"] = {}
                    
                    if retriever in item['retrieval_with_plans']:
                        continue
                    else:
                        retrieval_results = {}

                        hyp = item['hypothesis']
                        all_e = item['all_evidence']

                        retrieval_results["vanilla"] = retrieve(hyp, all_e, topk=len(all_e), model_name=retriever)

                        for planner_key in item["plans"]:
                            plans = item["plans"][planner_key]
                            # aggregate before retrieval
                            query = " ".join([hyp]+plans)
                            agg_before = retrieval_wrapper_func(query, all_e, topk=len(all_e), model_name=retriever)

                            # aggregate after retrieval
                            agg_after = [retrieval_wrapper_func(q, all_e, topk=len(all_e), model_name=retriever) for q in [hyp]+plans]

                            retrieval_results[planner_key] = {
                                "aggregate_before_retrieval": agg_before,
                                "aggregate_after_retrieval": agg_after
                            }

                        item['retrieval_with_plans'][retriever] = retrieval_results
                if i % 20 == 0:
                    write_json(plan_fn, datasets)
            write_json(plan_fn, datasets)
