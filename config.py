# 使用字典映射eval_type到文件名模式
EVAL_FILE_PATTERNS = {
    "safe": f"{eval_type}_{mode}_{num_prompts}",
    "empirical": f"{eval_type}_{mode}_{num_prompts}",
    "harmful": f"{eval_type}_{num_prompts}",
    "smoothing": f"{eval_type}_{num_prompts}",
    "grad_ec": f"{eval_type}_{num_prompts}",
    "greedy_ec": f"{eval_type}_{num_prompts}",
    "roc_curve": f"{eval_type}_{max_erase}"
}

# 使用不同的模型
MODEL_CONFIGS: dict[str, dict] = {
    "Llama-2": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        # commit_id :"main"        # to use the latest version
        commit_id: "08751db2aca9bf2f7f80d2e516117a53d7450235",  # to reproduce the results in our paper
    },
    "Llama-2-13B": {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "commit_id": None,
    },
    "Llama-3": {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "commit_id": None,
    },
    "GPT-3.5": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        # "commit_id":  "main",       # to use the latest version
        "commit_id": "08751db2aca9bf2f7f80d2e516117a53d7450235",  # to reproduce the results in our paper
    },
}



    