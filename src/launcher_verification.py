import subprocess
from tqdm import tqdm
from utils import get_runstr_combinations, load_json
from config import verification_LLM_models

device_info = load_json("tmp/device.json")
DEVICE_NAME = device_info['device_name']


all_runstrs = []

header = "python run_verification_ensemble.py --do_verification "


""" Verify """
fixed_configs = {
}


comb_configs = {
    "llm_idx": list(range(len(verification_LLM_models))),
    "dataset_idx": [0, 1, 2, 3]
}


all_runstrs += get_runstr_combinations(header, fixed_configs, comb_configs)


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
