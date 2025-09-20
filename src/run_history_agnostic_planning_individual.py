import os
import argparse
from tqdm import tqdm
from data import get_datasets
from utils import write_json, load_json
from baseline import get_planner, retrieve


result_dir = os.path.join("tmp", "planning_history_agnostic")
os.makedirs(result_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--planner_idx", type=int, default=0)
parser.add_argument("--retriever_idx", type=int, default=0)
parser.add_argument("--dataset_idx", type=int, default=0)
parser.add_argument("--do_plan", action="store_true")
parser.add_argument("--do_aggregate_plans", action="store_true")
parser.add_argument("--do_retrieve", action="store_true")
args = parser.parse_args()
print(vars(args))

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

dataset_names = [
    "entailmentbank",
    "wice",
    "hotpotqa",
    "musique"
]

retrievers = [
    "BM25",
    "sentence-transformers/sentence-t5-large",
    "all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large-instruct"
]

do_plan = True
# # do_plan = False
do_retrieve = True
# do_retrieve = False


if args.do_plan:
    # Generate all plans
    dataset = dataset_names[args.dataset_idx]
    planner_type, planner_model_name = planners[args.planner_idx]
    planner_key = planner_type + "|" + \
        (planner_model_name if planner_model_name else "")
    plan_fn = os.path.join(result_dir, f"plans_{planner_key}_{dataset}.json")

    if os.path.exists(plan_fn):
        datasets = load_json(plan_fn)
    else:
        datasets = get_datasets()

    print("Generating planning queries.")
    print(planner_key, dataset)

    planner = get_planner(planner_type, planner_model_name)
    data_split = dataset
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

if args.do_aggregate_plans:
    # Aggregate all plans results to a single plans.json
    print("Aggregating all plans to a single plans.json file.")
    print("Checking that all plan files exist. And add plans to aggregated file.")

    datasets = get_datasets()
    output_plan_fn = os.path.join(result_dir, f"plans.json")
    output_plan_retrieval_fn = os.path.join(result_dir, f"plans_retrieval.json")

    if not os.path.exists(output_plan_fn):
        print("Plan file not existing. Aggregating plans.")
        for planner_idx in tqdm(range(len(planners))):
            for dataset_idx in tqdm(range(len(dataset_names))):
                dataset = dataset_names[dataset_idx]
                planner_type, planner_model_name = planners[planner_idx]
                planner_key = planner_type + "|" + \
                    (planner_model_name if planner_model_name else "")
                plan_fn = os.path.join(
                    result_dir, f"plans_{planner_key}_{dataset}.json")

                assert os.path.exists(plan_fn)

                plans = load_json(plan_fn)

                # update
                for dataset_item, plan_item in zip(datasets[dataset], plans[dataset]):
                    if "plans" not in dataset_item:
                        dataset_item['plans'] = {}
                    dataset_item['plans'].update(plan_item.get('plans', {}))

        write_json(output_plan_fn, datasets)\
    
    else:
        datasets = load_json(output_plan_fn)
        
    if not os.path.exists(output_plan_retrieval_fn):
        print("Plan+retrieval results file not existing. Aggregating plans with retrieval results.")
        for retriever in tqdm(retrievers):
            
            partial_results_fn = os.path.join(
                result_dir, f"retrieval_{retriever.split("/")[-1]}.json")

            assert os.path.exists(partial_results_fn)

            partial_results = load_json(partial_results_fn)

            # update
            for dataset in dataset_names:
                for dataset_item, results_item in zip(datasets[dataset], partial_results[dataset]):
                    if 'retrieval_with_plans' not in dataset_item:
                        dataset_item['retrieval_with_plans'] = {}
                    dataset_item['retrieval_with_plans'].update(results_item.get('retrieval_with_plans', {}))

        write_json(output_plan_retrieval_fn, datasets)
    print("Done. Results saved.")


if args.do_retrieve:
    # Retrieve according to planned queries
    plan_fn = os.path.join(result_dir, f"plans.json")
    retriever = retrievers[args.retriever_idx]
    retrieval_fn = os.path.join(result_dir, f"retrieval_{retriever.split("/")[-1]}.json")

    assert os.path.exists(plan_fn) and not os.path.exists(retrieval_fn)
    datasets = load_json(plan_fn)

    def retrieval_wrapper_func(query, corpus, topk, model_name):
        res = retrieve(query, corpus, topk, model_name)
        res = {k: list(res[k]) for k in res}
        return {"similarity": res['similarity'], 'indices': res['indices']}


    for data_split in datasets:
        print(retriever, data_split)
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

                    retrieval_results["vanilla"] = retrieve(
                        hyp, all_e, topk=len(all_e), model_name=retriever)

                    for planner_key in item["plans"]:
                        plans = item["plans"][planner_key]
                        # aggregate before retrieval
                        query = " ".join([hyp]+plans)
                        agg_before = retrieval_wrapper_func(
                            query, all_e, topk=len(all_e), model_name=retriever)

                        # aggregate after retrieval
                        agg_after = [retrieval_wrapper_func(q, all_e, topk=len(
                            all_e), model_name=retriever) for q in [hyp]+plans]

                        retrieval_results[planner_key] = {
                            "aggregate_before_retrieval": agg_before,
                            "aggregate_after_retrieval": agg_after
                        }

                    item['retrieval_with_plans'][retriever] = retrieval_results
        write_json(retrieval_fn, datasets)
