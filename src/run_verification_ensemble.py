import os
import argparse
from data import get_datasets
from tqdm import tqdm
from baseline import ask_ensemble_two_way
from utils import write_json, load_json
from config import verification_LLM_models
from random import seed

""" Combining LLMs + NLIs for groundedness verification. """

parser = argparse.ArgumentParser()
parser.add_argument("--llm_idx", type=int, default=0)
# parser.add_argument("--retriever_idx", type=int, default=0)
parser.add_argument("--dataset_idx", type=int, default=0)
parser.add_argument("--do_verification", action="store_true")
parser.add_argument("--do_aggregate_ensemble_results", action="store_true")
args = parser.parse_args()
print(vars(args))

dataset_names = [
    "entailmentbank",
    "wice",
    "hotpotqa",
    "musique"
]

run_all = False     # whether to run all models on all four datasets at once
random_state = 20240627

nli_model_name = "microsoft/deberta-v2-xxlarge-mnli" # use this model's prediction for ensemble.
result_dir = os.path.join("tmp", "verification")

in_fn = os.path.join(result_dir, "verification_results.json")

datasets = get_datasets()

os.makedirs(result_dir, exist_ok=True)


# make test set

print('making evaluation test set for model')
seed(random_state)

verification_test = {}
assert os.path.exists(in_fn)
verification_test = load_json(in_fn)


if args.do_verification:
    print("[Running Verification Evaluation Experiments]")
    if run_all: 
        # Running results for all
        
        out_fn = os.path.join(result_dir, "ensemble_verification_results.json")
        for model in verification_LLM_models:
            print("Running results for:", model)
            for key in verification_test:
                print(key)
                for item in tqdm(verification_test[key]):
                    ticker = "ensemble|"+model
                    
                    if ticker in item:   # skip tested
                        continue
                    
                    retry = 0
                    while retry < 3:
                        try:
                            res = ask_ensemble_two_way(item['e1'], item['e2'], model=model, mode='two-way', nli_preds=item[nli_model_name])
                            item[ticker] = res
                            break

                        except Exception as e:
                            print(e)
                        retry += 1
                        
                    # break
                    
                write_json(out_fn, verification_test)

    else:
        model = verification_LLM_models[args.llm_idx]
        key = dataset_names[args.dataset_idx]
        
        out_fn = os.path.join(result_dir, f"ensemble_verification_results+{key}+{model.split("/")[-1]}.json")

        print("Running results for", model, key)
        for item in tqdm(verification_test[key]):
            ticker = "ensemble|"+model
            
            if ticker in item:   # skip tested
                continue
            
            retry = 0
            while retry < 3:
                try:
                    res = ask_ensemble_two_way(item['e1'], item['e2'], model=model, mode='two-way', nli_preds=item[nli_model_name])
                    item[ticker] = res
                    break

                except Exception as e:
                    print(e)
                retry += 1
                
            
        write_json(out_fn, verification_test)
        


if args.do_aggregate_ensemble_results:
    print("[Aggregating verificaiton results to a single verification_results_all.json]")
    # Aggregate all plans results to a single plans.json
    print("Checking that all partial results files exist. And add to aggregated file.")

    for llm_idx in tqdm(range(len(verification_LLM_models))):
        for dataset_idx in tqdm(range(len(dataset_names))):
            dataset = dataset_names[dataset_idx]
            model = verification_LLM_models[llm_idx]
            
            partial_results_fn = os.path.join(
                result_dir, f"ensemble_verification_results+{dataset}+{model.split("/")[-1]}.json")

            if not os.path.exists(partial_results_fn):
                # raise FileNotFoundError(partial_results_fn)
                print("Not Found:", partial_results_fn)
                continue

            partial_results = load_json(partial_results_fn)

            # update
            for dataset_item, results_item in zip(verification_test[dataset], partial_results[dataset]):
                dataset_item.update(results_item)

    write_json(os.path.join(result_dir, f"verification_results_all.json"), verification_test)
    print("Done. Results saved.")
