import os
import argparse
from tqdm import tqdm
from data import get_datasets
from utils import write_json, load_json, get_dict, get_llm_response
from baseline import retrieve

result_dir = os.path.join("tmp", "planning_history_aware")
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



# HISTORY AWARE PLANNER 
plan_with_history_prompt = """You are an AI information retrieval specialist trained to optimize search queries for finding relevant evidence in factual sources.

Task: Generate targeted search queries to find evidence that could either support or disprove the given hypothesis.

Requirements:
1. Generate 3-5 refined search queries
2. Each query should be specific and focused
3. Consider both supporting and contradicting evidence
4. You may retain effective queries from the previous round

Input Hypothesis: {query}

Previous Information:
- Previous queries: {history_plan}
- Previous search results: {history_retrieval_results}

Output Format:
{{"queries": ["QUERY TEXT 1", "QUERY TEXT 2", ...]}}"""


def plan_with_history(input_text, history_plan, history_retrieval_results, model="claude-v3.5-sonnet"):
    retry = 0
    while retry < 3:
        try:
            res = get_llm_response(plan_with_history_prompt.format(query=input_text, 
                                                                   history_plan=history_plan, 
                                                                   history_retrieval_results=history_retrieval_results), model=model)
            # res = res[res.index("-"):]
            res = eval(get_dict(res))['queries']
            break
        except Exception as e:
            print("[In history aware planning]", e, retry)
            retry += 1

    return res

# PARAMETERS

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
    # "all-MiniLM-L6-v2",
    # "intfloat/multilingual-e5-large-instruct"
]


if args.do_plan:
    # Generate all plans
    dataset = dataset_names[args.dataset_idx]
    planner_type, planner_model_name = planners[args.planner_idx]
    planner_key = planner_type + "|" + \
        (planner_model_name if planner_model_name else "")
    retriever = retrievers[args.retriever_idx]
    
    result_key = planner_key + "|" + retriever.split("/")[-1]
    
    # Last round plans & results loading
    input_plan_retrieval_results_fn = "/home/ubuntu/EventGrounding/new_situation_ground/saved_results/plans_retrieval.agnostic.json"
    output_fn = os.path.join(result_dir, f"plans_{result_key}_{dataset}.json")
    
    f"plans_{planner_key}_{dataset}.json"
    


    assert os.path.exists(input_plan_retrieval_results_fn)
    datasets = load_json(input_plan_retrieval_results_fn)
    for key in list(datasets.keys()):
        if key != dataset:
            del datasets[key]

    print("Generating planning queries.")
    print(result_key, dataset)


    data_split = dataset
    for i, item in enumerate(tqdm(datasets[data_split])):
        hypothesis = item['hypothesis']
        gt_evidence = item['gt_evidence']
        all_evidence = item['all_evidence']

        history_plan = item['plans'][planner_key]
        history_retrieval_results_ids = item['retrieval_with_plans'][retriever][planner_key]['aggregate_before_retrieval']['indices'][:3]
        history_retrieval_results = [all_evidence[idx] for idx in history_retrieval_results_ids]
        
        if "plans_with_history" not in item:
            item["plans_with_history"] = {}

        if result_key in item["plans_with_history"]:
            continue
        else:
            plans = plan_with_history(hypothesis, history_plan, history_retrieval_results, model=planner_model_name)
            item["plans_with_history"][result_key] = plans

    write_json(output_fn, datasets)

if args.do_aggregate_plans:
    # Aggregate all plans results to a single plans.json
    print("Aggregating all plans to a single plans.json file.")
    print("Checking that all plan files exist. And add plans to aggregated file.")

    datasets = get_datasets()
    output_plan_fn = os.path.join(result_dir, f"plans_with_history.json")
    output_plan_retrieval_fn = os.path.join(result_dir, f"plans_with_history_retrieval.json")

    if not os.path.exists(output_plan_fn):
        print("Plan file not existing. Aggregating plans.")
        for planner_idx in tqdm(range(len(planners))):
            for retriever_idx in tqdm(range(len(retrievers))):
                for dataset_idx in tqdm(range(len(dataset_names))):
                    dataset = dataset_names[dataset_idx]
                    planner_type, planner_model_name = planners[planner_idx]
                    planner_key = planner_type + "|" + \
                        (planner_model_name if planner_model_name else "")
                    retriever = retrievers[retriever_idx]
                    
                    result_key = planner_key + "|" + retriever.split("/")[-1]
                        
                    plan_fn = os.path.join(result_dir, f"plans_{result_key}_{dataset}.json")

                    assert os.path.exists(plan_fn)

                    plans = load_json(plan_fn)

                    # update
                    for dataset_item, plan_item in zip(datasets[dataset], plans[dataset]):
                        if "plans_with_history" not in dataset_item:
                            dataset_item["plans_with_history"] = {}
                        dataset_item["plans_with_history"].update(plan_item.get("plans_with_history", {}))

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
                    if "retrieval_with_plans_history" not in dataset_item:
                        dataset_item["retrieval_with_plans_history"] = {}
                    dataset_item["retrieval_with_plans_history"].update(results_item.get("retrieval_with_plans_history", {}))

        write_json(output_plan_retrieval_fn, datasets)
    print("Done. Results saved.")


if args.do_retrieve:
    # Retrieve according to planned queries
    plan_fn = os.path.join(result_dir, f"plans_with_history.json")
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
            if "plans_with_history" in item:
                if "retrieval_with_plans_history" not in item:
                    item["retrieval_with_plans_history"] = {}

                if retriever in item["retrieval_with_plans_history"]:
                    continue
                else:
                    retrieval_results = {}

                    hyp = item['hypothesis']
                    all_e = item['all_evidence']

                    # retrieval_results["vanilla"] = retrieve(
                    #     hyp, all_e, topk=len(all_e), model_name=retriever)

                    for result_key in item["plans_with_history"]:
                        
                        if retriever.split("/")[-1] not in result_key:
                            continue
                        
                        plans = item["plans_with_history"][result_key]
                        # aggregate before retrieval
                        if isinstance(plans, str): plans = [plans]
                        query = " ".join([hyp]+plans)
                        agg_before = retrieval_wrapper_func(
                            query, all_e, topk=len(all_e), model_name=retriever)

                        # # aggregate after retrieval
                        # agg_after = [retrieval_wrapper_func(q, all_e, topk=len(
                        #     all_e), model_name=retriever) for q in [hyp]+plans]

                        retrieval_results[result_key] = {
                            "aggregate_before_retrieval": agg_before,
                            # "aggregate_after_retrieval": agg_after
                        }

                    item["retrieval_with_plans_history"][retriever] = retrieval_results
        write_json(retrieval_fn, datasets)
