import subprocess
from tqdm import tqdm
from utils import get_runstr_combinations, load_json

device_info = load_json("tmp/device.json")
DEVICE_NAME = device_info['device_name']


all_runstrs = []


""" Plan (history agnostic) """
# header = "python run_history_agnostic_planning_individual.py --do_plan"

# fixed_configs = {
# }


# comb_configs = {
#     "planner_idx": [0, 1, 2, 3, 4, 5, 6, 7],
#     "dataset_idx": [0, 1, 2, 3]
# }


# all_runstrs += get_runstr_combinations(header, fixed_configs, comb_configs)


""" Plan (history aware) """
header = "python run_history_aware_planning_individual.py --do_plan"

fixed_configs = {
}


comb_configs = {
    "planner_idx": [0, 1, 2, 3, 4, 5, 6, 7],
    "dataset_idx": [0, 1, 2, 3],
    "retriever_idx": [0, 1]
}


all_runstrs += get_runstr_combinations(header, fixed_configs, comb_configs)


""" Retrieve """

# pueue add -- "sleep 1800 && python run_history_agnostic_planning_individual.py --do_aggregate_plans"
# pueue add -- "python run_history_agnostic_planning_individual.py --do_retrieve --retriever_idx 0"
# pueue add -- "python run_history_agnostic_planning_individual.py --do_retrieve --retriever_idx 1"
# pueue add -- "python run_history_agnostic_planning_individual.py --do_retrieve --retriever_idx 2"
# pueue add -- "python run_history_agnostic_planning_individual.py --do_retrieve --retriever_idx 3"

if __name__ == "__main__":

    for i, runstr in enumerate(tqdm(all_runstrs)):
        print(runstr)

    # Submit commands to task scheduler
    for i, runstr in enumerate(tqdm(all_runstrs)):
        ticker = f"[GR-{DEVICE_NAME}-{i+1}/{len(all_runstrs)}]"

        queued_runstr = "pueue add \"{}\"".format(
            runstr + " && " +
            'python notify.py --subject {}'.format(ticker+"_END")
        )

        subprocess.check_call(queued_runstr, shell=True)
