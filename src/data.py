import os
import json
from tqdm import tqdm
from datasets import load_dataset
from utils import load_jsonl, load_json, write_json, get_llm_response, get_dict
from random import seed, sample

qa_to_prop_prompt = \
    """Paraphrase the given question and answer pair to a proposition. Your response should be formatted as {{"Proposition": "PROPOSITION TEXT"}}.

Question: When did the maker of the Acura Legend, the manufacturer of Toyopet Master, and Nissan open US assembly plants?
Answer: 1981
{{"Proposition": "The maker of the Acura Legend, the manufacturer of Toyopet Master, and Nissan opened US assembly plants in 1981."}}

Question: Signed with Maybach Music Group in 2011, which artist was featured as a guest in Fire of Zamani?
Answer: Wale
{{"Proposition": "Wale, who signed with Maybach Music Group in 2011, was a featured guest artist on Fire of Zamani."}}

Question: {question}
Answer: {answer}"""


def get_prop_from_qa(question, answer, model="gpt4"):
    response_dict_str = get_dict(get_llm_response(
        qa_to_prop_prompt.format(question=question, answer=answer), model=model))
    return json.loads(response_dict_str)['Proposition']


def get_musique_data(split_name='validation', max_evidence_num=200, min_gt_num=3, sample_n=500,
                     folder="/home/ubuntu/data/data/MuSiQue/data", qa_to_prop_model="gpt4"):
    if split_name == 'validation':
        split_name = "dev"
    elif split_name == 'test':
        pass
    elif split_name == 'train':
        pass
    else:
        raise ValueError(split_name+": Not supported value!")

    # Path to the QA->proposition file
    qa_to_prop_fn = os.path.join(
        folder, "{}_{}_qa_to_prop.json".format(split_name, qa_to_prop_model))
    if os.path.exists(qa_to_prop_fn):
        qa_to_prop = load_json(qa_to_prop_fn)
    else:
        qa_to_prop = {}

    fn = os.path.join(folder, "musique_ans_v1.0_{}.jsonl".format(split_name))
    data = load_jsonl(fn)

    # Preprocess and sample
    new_data = []
    for item in data:
        all_paragraphs = item['paragraphs']
        all_evidence = [i['paragraph_text'] +
                        "\nTitle \"{}\"".format(i['title']) for i in all_paragraphs]
        gts = [i['paragraph_support_idx']
               for i in item['question_decomposition']]

        question = item['question']
        answer = item['answer']

        # Filter by maximum total evidence num and minimum gt evidence num
        if len(all_paragraphs) > max_evidence_num:
            continue
        if len(gts) < min_gt_num:
            continue

        new_data.append({
            # "hypothesis": proposition,
            "gt_evidence": [all_evidence[i] for i in gts],
            "all_evidence": all_evidence,
            "gt_ids": list(sorted(gts)),
            "label": "entailment",
            "meta": {"id": item['id'], "question": question, "answer": answer, "qa_to_prop_model": qa_to_prop_model, "question_decomposition": item['question_decomposition']}
        })

    # Sample a subset if necessary
    if len(new_data) > sample_n:
        seed(2024)
        new_data = sample(new_data, sample_n)

    data = new_data

    # Convert QA to hypothesis
    for item in tqdm(data):
        question = item['meta']['question']
        answer = item['meta']['answer']
        qa = f"Q: {question} \nA: {answer}"
        if qa in qa_to_prop:
            proposition = qa_to_prop[qa]
        else:
            # prompting
            retry = 0
            while retry < 3:
                try:
                    proposition = get_prop_from_qa(
                        question, answer, model=qa_to_prop_model)
                    qa_to_prop[qa] = proposition
                    # save qa_to_prop
                    write_json(qa_to_prop_fn, qa_to_prop)
                    break
                except Exception as e:
                    print(e)
                    retry += 1

        item['hypothesis'] = proposition

    # save qa_to_prop
    write_json(qa_to_prop_fn, qa_to_prop)

    return data


def get_hotpotqa_data(split_name='validation', max_evidence_num=200, min_gt_num=3, sample_n=500,
                      folder="/home/ubuntu/data/data/hotpotqa", qa_to_prop_model="gpt4"):
    if split_name == 'validation':
        pass
    elif split_name == 'test':
        pass
    elif split_name == 'train':
        pass
    else:
        raise ValueError(split_name+": Not supported value!")

    # Path to the QA->proposition file
    qa_to_prop_fn = os.path.join(
        folder, "{}_{}_qa_to_prop.json".format(split_name, qa_to_prop_model))
    if os.path.exists(qa_to_prop_fn):
        qa_to_prop = load_json(qa_to_prop_fn)
    else:
        qa_to_prop = {}

    data = list(load_dataset("hotpotqa/hotpot_qa", "distractor")[split_name])

    # Preprocess and sample
    new_data = []
    for item in data:
        try:
            all_paragraphs = dict(sum([[((title, id_), sent) for id_, sent in sents] for title, sents in zip(
                item['context']['title'], [list(enumerate(para)) for para in item['context']['sentences']])], []))
            all_evidence = [text.strip()+"\nTitle: \"{}\"".format(title) for texts,
                            title in zip(item['context']['sentences'], item['context']['title']) for text in texts]

            all_evidence_texts = [text for text in sum(
                item['context']['sentences'], [])]
            gt_evidence = [all_paragraphs[tup] for tup in zip(
                item['supporting_facts']['title'], item['supporting_facts']['sent_id'])]
            gts = [all_evidence_texts.index(evidence)
                   for evidence in gt_evidence]

            question = item['question']
            answer = item['answer']

            # Filter by maximum total evidence num and minimum gt evidence num
            if len(all_paragraphs) > max_evidence_num:
                continue

            if len(gts) < min_gt_num:
                continue

            new_data.append({
                # "hypothesis": proposition,
                "gt_evidence": [all_evidence[gt_id] for gt_id in gts],
                "all_evidence": all_evidence,
                "gt_ids": list(sorted(gts)),
                "label": "entailment",
                "meta": {"id": item['id'], "question": question, "answer": answer, "type": item['type'], "qa_to_prop_model": qa_to_prop_model}
            })
        except Exception as e:
            print(e)
            continue

    if len(new_data) > sample_n:
        # sample a subset
        seed(2024)
        new_data = sample(new_data, sample_n)

    data = new_data

    # Convert QA to hypothesis
    for item in tqdm(data):

        question = item['meta']['question']
        answer = item['meta']['answer']
        qa = f"Q: {question} \nA: {answer}"
        if qa in qa_to_prop:
            proposition = qa_to_prop[qa]
        else:
            # prompting
            retry = 0
            while retry < 3:
                try:
                    proposition = get_prop_from_qa(
                        question, answer, model=qa_to_prop_model)
                    qa_to_prop[qa] = proposition
                    # save qa_to_prop
                    write_json(qa_to_prop_fn, qa_to_prop)
                    break
                except Exception as e:
                    print(e)
                    retry += 1

        item['hypothesis'] = proposition

    # save qa_to_prop
    write_json(qa_to_prop_fn, qa_to_prop)

    return data


def get_wice_data(split_name='validation', level="subclaim", max_evidence_num=200, min_gt_num=2,
                  folder="/home/ubuntu/data/data/wice/wice/data/entailment_retrieval"):
    if split_name == 'validation':
        split_name = "dev"
    elif split_name == 'test':
        pass
    elif split_name == 'train':
        pass
    else:
        raise ValueError(split_name+": Not supported value!")

    data = load_jsonl(os.path.join(folder, level, f"{split_name}.jsonl"))

    all_data = []
    for item in data:
        # Filter by maximum total evidence num and minimum gt evidence num
        if len(item['evidence']) > max_evidence_num:
            continue

        if any([len(gts) < min_gt_num or len(gts) > 8 for gts in item['supporting_sentences']]):
            continue

        if item['label'] != "supported":
            continue

        all_gt_evidence = [[item['evidence'][id_] for id_ in group]
                           for group in item['supporting_sentences']]
        all_gt_ids = item['supporting_sentences']
        gt_evidence = all_gt_evidence[0]
        gt_ids = all_gt_ids[0]

        all_data.append({
            "hypothesis": item['claim'],
            "gt_evidence": gt_evidence,
            "all_evidence": item['evidence'],
            "gt_ids": gt_ids,
            "label": item['label'],
            "meta": item['meta']
        })

    return all_data


def get_entailmentbank_data(split_name='validation', level=2, max_evidence_num=200, min_gt_num=2,
                            data_dir="/home/ubuntu/data/data/EntailmentBank"):
    if split_name == 'validation':
        split_name = "dev"
    elif split_name == 'test':
        pass
    elif split_name == 'train':
        pass
    else:
        raise ValueError(split_name+": Not supported value!")

    # EntailmentBank T2 (GT + distractors)
    if level == 2:
        bank = load_jsonl(os.path.join(
            data_dir, "v3_May6_2022/entailment_trees_emnlp2021_data_v3/dataset/", f'task_2/{split_name}.jsonl'))
    elif level == 3:
        bank = load_jsonl(os.path.join(
            data_dir, "v3_May6_2022/entailment_trees_emnlp2021_data_v3/dataset/", f'task_3/{split_name}.jsonl'))

    all_data = []
    for item in bank:
        all_propositions = item['meta']['triples']
        not_gt_ids = item['meta']['distractors']
        gt_ids = [id_ for id_ in all_propositions if id_ not in not_gt_ids]
        gt_propositions = [all_propositions[id_] for id_ in gt_ids]
        all_propositions = [all_propositions[id_] for id_ in all_propositions]

        # Filter by maximum total evidence num and minimum gt evidence num
        if len(all_propositions) > max_evidence_num:
            continue
        if any([len(gts) < min_gt_num for gts in gt_ids]):
            continue

        all_data.append({
            "hypothesis": item['hypothesis'],
            "gt_evidence": gt_propositions,
            "all_evidence": all_propositions,
            "gt_ids": [all_propositions.index(p) for p in gt_propositions],
            "label": "entailment",
            "meta": {
                "id": item['id'],
                "depth_of_proof": item["depth_of_proof"]
            }
        })

    return all_data


def get_propsegment_data(split_name='validation'):
    propsegment = load_dataset("sihaochen/propsegment", "nli")

    split = propsegment[split_name]
    all_hypothesis = [i['hypothesis'] for i in split]
    all_orig_hypothesis = [i.replace("[M]", "").replace(
        "[/M]", "") for i in all_hypothesis]

    all_data = {}
    for item, hypo, orig_hypo in zip(split, all_hypothesis, all_orig_hypothesis):
        if orig_hypo not in all_data:
            all_data[orig_hypo] = {"hypothesis": orig_hypo,
                                   "propositions": [], "premise_list": {}}

        if item['premise'] not in all_data[orig_hypo]['premise_list']:
            if len(all_data[orig_hypo]['premise_list']) == 0:
                id_ = 0
            else:
                id_ = max(all_data[orig_hypo]['premise_list'].values()) + 1

            all_data[orig_hypo]['premise_list'][item['premise']] = id_

        all_data[orig_hypo]['propositions'].append({
            "proposition": hypo,
            "premise": all_data[orig_hypo]['premise_list'][item['premise']],
            "label": item['label']
        })

    # reformat into a list of dict
    all_data = [all_data[key] for key in all_data]

    return all_data


def get_datasets():
    entailmentbank = get_entailmentbank_data(
        split_name="test",  level=2, max_evidence_num=200, min_gt_num=2)

    wice_claim = get_wice_data(split_name="train", level="claim",
                               max_evidence_num=200, min_gt_num=2)  # Use the training set since (1) the amount of qualified instances in validation / test set is too low; and (2) the dataset is relatively new so that it is likely not leaked to LLMs' training.

    hotpotqa = get_hotpotqa_data(split_name="validation", min_gt_num=3,
                                 sample_n=500, qa_to_prop_model="gpt4")  # only validation set available
    musique = get_musique_data(
        split_name="validation", min_gt_num=3, sample_n=500, qa_to_prop_model="gpt4")

    datasets = {
        "entailmentbank": entailmentbank,
        "wice": wice_claim,
        "hotpotqa": hotpotqa,
        "musique": musique
    }
    return datasets
