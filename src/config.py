device = 'cuda:0'

# dataset parameters
max_evidence_num = 200

# retrievers
retriever_models = {"Random": "Random",
        "BM25": "BM25",
          "princeton-nlp/sup-simcse-roberta-base": r"SimCSE$_{\text{RoBERTa-base}}$", 
          "all-MiniLM-L6-v2": "MiniLM-L6",  
          "sentence-transformers/sentence-t5-large": r"ST5$_{\text{large}}$",
          'sentence-transformers/gtr-t5-large': r"GTR$_{\text{T5-large}}$",
          "intfloat/multilingual-e5-large-instruct": r"mE5$_{\text{large-instruct}}$",

}

instruct_retrievers = ["intfloat/multilingual-e5-large-instruct"]
retrieval_instructions = {
    "|default": 'Given a web search query, retrieve relevant passages that answer the query',
    "|hypothesis-premise": "Given a hypothesis (query), retrieve all necessary premises that entail / contradict the hypothesis."
}

retrieval_metrics = ["ndcg@5", "ndcg@10",
           "recall@5", "recall@10", 
            "precision@5", "precision@10", 
            "f1@5", "f1@10", 
            # "r-precision", 
            # "hit_rate@5", "hit_rate@10",
        ]

# NLI models 
verification_NLI_models = [
    "microsoft/deberta-v2-xxlarge-mnli",
    "microsoft/deberta-xlarge-mnli"
]

verification_LLM_models = [
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "claude-v3-haiku",
    "claude-v3-sonnet",
    "claude-v3.5-sonnet"
]

verification_models = verification_NLI_models + verification_LLM_models

model_name_map = {
    "microsoft/deberta-v2-xxlarge-mnli": "NLI-xxlarge",
    "microsoft/deberta-xlarge-mnli": "NLI-xlarge",
    "llama3.1-8b-instruct": "Llama3.1 8B Instr.",
    "llama3.1-70b-instruct": "Llama3.1 70B Instr." ,
    "claude-v3-haiku": "Claude3 Haiku",
    "claude-v3-sonnet": "Claude3 Sonnet",
    "claude-v3.5-sonnet": "Claude3.5 Sonnet"
}

ensemble_model_name_map = {
    f"ensemble|{key}": f"ensemble|{val}" for key, val in model_name_map.items()
}

dataset_names = [
    "entailmentbank",
    "wice",
    "hotpotqa",
    "musique"
]