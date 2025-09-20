import json
import torch
from utils import get_dict
from copy import deepcopy
from external.query_expansion import query_expansion
from external.factscore import factscore_decompose
from external.propositionizer import proposition_decompose
from external.entailer import entailer_generate
from config import device, verification_NLI_models
from transformers import pipeline
from utils import get_llm_response, get_dict
from config import device

""" Retrieval methods """


def random_retrieve(query, corpus, topk=5):
    from random import shuffle
    indices = list(range(len(corpus)))
    shuffle(indices)
    return {"result": [corpus[i] for i in indices][:topk],
            "similarity": list(reversed([float(sim) for sim in range(len(corpus))]))[:topk],
            "indices": indices[:topk]}


def BM25_retrieve(query, corpus, topk=5):
    from rank_bm25 import BM25Okapi
    from nltk import word_tokenize

    tokenized_query = word_tokenize(query.lower())
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    doc_scores = bm25.get_scores(tokenized_query)
    result = bm25.get_top_n(tokenized_query, range(len(corpus)), n=topk)
    return {"result": [corpus[i] for i in result],
            "similarity": list(sorted(doc_scores, reverse=True)),
            "indices": result}


def langchain_BM25_retrieve(query, corpus, topk=5):
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_texts(corpus, k=topk)
    return {"result": retriever.invoke(query)}


def embed_retrieve(query, corpus, topk=5, model_name="all-MiniLM-L6-v2"):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.utils.math import cosine_similarity_top_k
    # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    global embeddings

    reload = False
    if "embeddings" in globals():
        embeddings = globals()["embeddings"]
        if embeddings.model_name != model_name:
            print(embeddings.model_name, model_name)
            reload = True
    else:
        print("embeddings not in vars")
        reload = True

    if reload:
        print("Loading embedding model:", model_name)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={'device': device},)

    query_result = embeddings.embed_query(query)
    doc_result = embeddings.embed_documents(corpus)

    indices, similarities = cosine_similarity_top_k(
        [query_result], doc_result, top_k=topk, score_threshold=None,)
    indices = [int(i[1]) for i in indices]

    return {"result": [corpus[i] for i in indices], "similarity": similarities, "indices": indices}


def retrieve(query, corpus, topk=5, model_name="BM25", task_description="Given a hypothesis (query), retrieve all necessary premises that entail / contradict the hypothesis."):
    if model_name == "BM25":
        return BM25_retrieve(query, corpus, topk)

    elif model_name == "Random":
        return random_retrieve(query, corpus, topk)

    elif model_name == "intfloat/multilingual-e5-large-instruct":
        # instruct embeddings
        # task instructions can be like: 'Given a web search query, retrieve relevant passages that answer the query'
        def get_detailed_instruct(task_description: str, query: str) -> str:
            # From https://huggingface.co/intfloat/multilingual-e5-large-instruct
            return f'Instruct: {task_description}\nQuery: {query}'

        assert isinstance(task_description, str)
        query = get_detailed_instruct(task_description, query)
        return embed_retrieve(query, corpus, topk, model_name)

    else:
        return embed_retrieve(query, corpus, topk, model_name)


""" Verification methods """


def init_nli_classifier(model="microsoft/deberta-v2-xxlarge-mnli", device='cuda:0'):
    global nli_classifier

    nli_classifier = pipeline("text-classification",
                              model=model, device=device)


def nli(e1, e2, model="microsoft/deberta-v2-xxlarge-mnli", device='cuda:0'):
    global nli_classifier

    if 'nli_classifier' not in globals():
        print("Initing model:", model)
        init_nli_classifier(model=model, device=device)

    elif globals()['nli_classifier'].model.name_or_path != model:
        del nli_classifier
        torch.cuda.empty_cache()
        print("Initing model:", model)
        init_nli_classifier(model=model, device=device)

    return nli_classifier({'text': e1, 'text_pair': e2}, top_k=3)


def nli_ask(e1, e2, model="microsoft/deberta-v2-xxlarge-mnli"):
    if isinstance(e1, list):
        e1 = " ".join(e1)
    return nli(e1, e2, model, device)


def llm_ask(e1, e2, model="llama3-8b-instruct", candidates=["entailment", "contradiction", "neutral"]):
    labels = " or ".join(candidates)
    KB_ask_prompt = (
        f"You are a helpful logical reasoner. Please help classify a hypothesis with {labels} based solely on a set of evidence.\n"
        f"Evidence set: {e1}\n"
        f"Hypothesis: {e2}\n"
        f'Result in JSON format (e.g. {{"label": "{labels}"}}):'
    )
    response = get_llm_response(
        KB_ask_prompt, model=model, max_new_tokens=1024)
    return json.loads(get_dict(response))


def ask(e1, e2, model="llama3-8b-instruct", mode="two-way"):
    """ Unified ask function for both llm and nli models. 
    e1: list of strings (evidence set)
    e2: string (hypothesis)
    mode: "two-way" (Entailment or not); "three-way" (Entailment / Contradiction / Neutral)
    """
    if model in verification_NLI_models:
        model_type = "nli"

        res = nli_ask(e1, e2, model)
        # post proc
        if mode == "two-way":
            is_entailment = res[0]['label'] == "ENTAILMENT"
            pred = "entailment" if is_entailment else "not entailment"
        elif mode == "three-way":  # three-way
            pred = res[0]['label'].lower()
        else:
            raise ValueError("Wrong mode: "+mode)

    else:
        model_type = "llm"

        if mode == "two-way":
            candidates = ["entailment", "not entailment"]
        elif mode == "three-way":  # three-way
            candidates = ["entailment", "contradiction", "neutral"]
        else:
            raise ValueError("Wrong mode: "+mode)

        res = llm_ask(e1, e2, model, candidates)
        pred = res['label']
        assert pred in candidates

    info = {
        "prediction": pred,
        "raw_prediction": res,
        "model": model,
        "model_type": model_type,
        "pred_mode": mode
    }
    return info


" For combining NLI & LLM outputs "


def llm_nli_ask(e1, e2, model="llama3-8b-instruct", candidates=["entailment", "contradiction", "neutral"], nli_label=None):
    labels = " or ".join(candidates)
    KB_ask_prompt = (
        f"You are a helpful logical reasoner. Please help classify a hypothesis with {labels} based solely on a set of evidence.\n"
        f"For your reference, an external supervised Natural Language Inference model's prediction is: {nli_label}. \n"
        f"Evidence set: {e1}\n"
        f"Hypothesis: {e2}\n"
        f'Result in JSON format (e.g. {{"label": "{labels}"}}):'
    )
    # print(KB_ask_prompt)
    response = get_llm_response(
        KB_ask_prompt, model=model, max_new_tokens=1024)
    return json.loads(get_dict(response))


def ask_ensemble_two_way(e1, e2, model="llama3-8b-instruct", mode="two-way", nli_preds=None):
    """ Unified ask function for both llm and nli models. 
    e1: list of strings (evidence set)
    e2: string (hypothesis)
    mode: "two-way" (Entailment or not); "three-way" (Entailment / Contradiction / Neutral)
    """
    if model in verification_NLI_models:
        raise ValueError(
            "Wrong model type: NLI models in ensemble mode. "+mode)

    else:
        model_type = "llm"

        if mode == "two-way":
            candidates = ["entailment", "not entailment"]
        else:
            raise ValueError("Wrong mode: "+mode)

        res = llm_nli_ask(e1, e2, model, candidates,
                          nli_label=nli_preds['prediction'])
        pred = res['label']
        assert pred in candidates

    info = {
        "prediction": pred,
        "raw_prediction": res,
        "model": model,
        "model_type": model_type,
        "pred_mode": mode
    }
    return info


""" Grounding models """


def get_planner(planner_type, planner_model_name=None, device="cuda:0"):
    """ Initialize and return the specified planner. 
    Supported planner_type (planner_model_name): entailer (allenai/entailer-large & allenai/entailer-11b), propositionizer, factscore (all LLMs), query_expansion (all LLMs)
    """
    from functools import partial

    if planner_type == "entailer":
        if planner_model_name is None or planner_model_name == "allenai/entailer-large":
            return partial(entailer_generate, model="allenai/entailer-large", device=device)
        else:  # "allenai/entailer-11b" or LLM tickers
            return partial(entailer_generate, model=planner_model_name, device=device)

    elif planner_type == "propositionizer":
        if planner_model_name is None:
            return partial(proposition_decompose, device=device)
        else:   # LLMs
            return partial(proposition_decompose, model=planner_model_name, device=device)

    elif planner_type == "factscore":
        if planner_model_name is None or planner_model_name == "claude-v3-sonnet":
            return partial(factscore_decompose, model="claude-v3-sonnet")
        else:
            return partial(factscore_decompose, model=planner_model_name)

    elif planner_type == "query_expansion":
        if planner_model_name is None or planner_model_name == "claude-v3-sonnet":
            return partial(query_expansion, model="claude-v3-sonnet")
        else:
            return partial(query_expansion, model=planner_model_name)

    else:
        raise ValueError("Not supported planner type: {}".format(planner_type))


plan_prompt = (
    "You are trying to find grounding information entries from a knowledge base for a hypothesis. \n"
    "Hypothesis: {hyp}\n"
    "Retrieved information: {info}\n\n"
    "Since the retrieval results are insufficient to ground the hypothesis, please generate potential proof using decompositions, negations over the hypothesis.\n"
    "E.g., [\"Socrates is mortal.\"] can be proved if [\"All men are mortal\", \"Socrates is a man\"] are proved.\n"
    "Your answer should be in the format similar to {{\"Hypotheses\": [\"Text1\", \"Text2\", ...]}}\n"
    "Answer:"
)

plan_prompt_strict = (
    "You are trying to find grounding information entries from a knowledge base for a hypothesis. \n"
    "Since the retrieval results are insufficient to ground the hypothesis, please generate potential proof using decompositions, negations over the hypothesis.\n"
    "The generated hypotheses should together entail the given hypothesis while take into account the retrieved information.\n"
    "Example 1:\n"
    "Hypothesis: [\"Socrates is mortal.\"]\n"
    "Retrieved information: [\"All men are mortal\"]\n"
    "Answer: {{\"Hypotheses\": [\"Socrates is a man\", \"All men are mortal\"]}}\n\n"
    "Example 2:\n"
    "Hypothesis: [\"people from new york state will see the sun rise in the east.\"]\n"
    "Retrieved information: [\"new york state is on earth.\"]\n"
    "Answer: {{\"Hypotheses\": [\"the sun rises in the east for people on earth.\", \"new york state is on earth.\"]}}\n\n"
    "Example 3:\n"
    "Hypothesis: {hyp}\n"
    "Retrieved information: {info}\n\n"
    "Answer:"
)


def plan(Phis, Sigmas, model="claude-v3-sonnet"):
    """ Generate proof structure based on last step. """
    response = get_llm_response(plan_prompt_strict.format(
        hyp=Phis[-1], info=Sigmas[-1]), model=model)
    return json.loads(get_dict(response))['Hypotheses']


def ground(hypothesis, K, T=2,
           retriever="BM25", topk=3,
           verifier='microsoft/deberta-v2-xxlarge-mnli', mode="two-way",
           planner="claude-v3-haiku"):
    """ Main implementation of the grounding method. """
    # Initialization
    Sigmas = []
    Phis = [hypothesis]
    all_verifs = []

    # Round 0
    retrieval_res = retrieve(hypothesis, K, topk=topk, model_name=retriever)
    hat_Sigma = retrieval_res["result"]
    verif_res = ask(hat_Sigma, hypothesis, model=verifier, mode=mode)
    Sigmas.append(hat_Sigma)

    all_verifs.append(verif_res)

    if verif_res["prediction"] == "entailment":
        return Sigmas[1]

    # Main rounds
    t = 1
    while t <= T and verif_res["prediction"] != "entailment":
        Phi = plan(Phis, Sigmas, model=planner)
        Phis.append(Phi)

        # Retrieve according to the new Phi
        hat_Sigma = deepcopy(Sigmas[-1])

        this_K = [k for k in K if k not in hat_Sigma]

        for h in Phi:
            ret_res = retrieve(h, this_K, topk=1, model_name=retriever)
            if ret_res["result"][0] not in hat_Sigma:
                hat_Sigma.append(ret_res["result"][0])

        # Verify
        verif_res = ask(hat_Sigma, hypothesis, model=verifier, mode=mode)
        Sigmas.append(hat_Sigma)
        all_verifs.append(verif_res)

        t += 1
    return Phis, Sigmas, all_verifs
