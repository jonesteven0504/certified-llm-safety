import torch
import transformers
from transformers import AutoTokenizer
from models import *
import os
import time
import json
import ast
import random
import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from math import ceil
import numpy as np

from defenses import is_harmful
from defenses import progress_bar, erase_and_check, erase_and_check_smoothing
from grad_ec import grad_ec
from greedy_ec import greedy_ec
from greedy_grad_ec import greedy_grad_ec

from openai import OpenAI
from config import MODEL_CONFIGS, EVAL_FILE_PATTERNS
from bashargs_parse_config import create_parser, print_eval_type, calculate_time,check_is_harmful_evaltype_safe, check_is_harmful_evaltype_empirical, check_is_harmful_evaltype_grad_ec, check_is_harmful_evaltype_greedy_ec, choose_prompts

# Step 1: parse args
# parser = argparse.ArgumentParser(description="Check safety of prompts.")
# 创建parser实例并进行parse解析
parser = create_parser()
args = parser.parse_args()

# # 直接通过 args.<参数名> 来访问参数
# print(f"Number of prompts: {args.num_prompts}")
# print(f"Eval type: {args.eval_type}")
# print(f"Use classifier: {args.use_classifier}")

num_prompts = args.num_prompts
mode = args.mode
eval_type = args.eval_type
max_erase = args.max_erase
num_adv = args.num_adv
results_dir = args.results_dir
use_classifier = args.use_classifier
model_wt_path = args.model_wt_path
safe_prompts_file = args.safe_prompts
harmful_prompts_file = args.harmful_prompts
randomize = args.randomize
sampling_ratio = args.sampling_ratio
num_iters = args.num_iters
llm_name = args.llm_name
attack = args.attack
ec_variant = args.ec_variant
adv_prompts_dir = args.adv_prompts_dir


# Part X： 配置实验结果的输出格式
print("\n* * * * * * Experiment Details * * * * * *")
if torch.cuda.is_available():
    print("Device: " + torch.cuda.get_device_name(0))
print("Evaluation type: " + eval_type)
print("Number of prompts to check: " + str(num_prompts))
# print("Append adversarial prompts? " + str(args.append_adv))
print("Use randomization? " + str(randomize))


if randomize:
    print("Sampling ratio: ", str(sampling_ratio))

if use_classifier:
    print("Using custom safety filter. Model weights path: " + model_wt_path)
else:
    print("Using LLM model: " + llm_name)



# 输出模型评价相关的配置信息
print_eval_type(eval_type)


# Step X： 创建文件夹
# Create results directory if it doesn't exist
# if use_classifier:
#     results_dir = results_dir + "_clf"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create results file

# 根据eval_type,从字典获取文件名模式
results_file_pattern = EVAL_FILE_PATTERNS.get(eval_type, eval_type)
results_file = os.path.join(results_dir, f"{results_file_pattern}.json")

# Add tag for safety classifier and randomized check
suffix = []
if use_classifier:
    suffix.append("clf")
if randomize:
    suffix.append("rand")

# 一次性统一处理替换操作，避免多次替换出错，同时增强可维护性（替换逻辑单独放）
if suffix:
    results_file = results_file.replace(".json", f"_{'_'.join(suffix)}.json")

# Load results if they exist
try:
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
except FileNotFoundError:
    results = {}


# Step X： 配置并加载大模型
if use_classifier:
    # Using custom classifier for safety filter
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    # Load model weights
    # path = 'models/distillbert_saved_weights.pt'
    model.load_state_dict(torch.load(model_wt_path))
    model.eval()

    # Create a text classification pipeline
    pipeline = transformers.pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=0
    )
else:
    # Using LLM for safety filter
    if llm_name in MODEL_CONFIGS:
        if llm_name == "GPT-3.5":

            with open("key.txt", "r") as file:
                key = file.read()
            pipeline = OpenAI(api_key=key)
            tokenizer = AutoTokenizer.from_pretrained(
                model=MODEL_CONFIGS[llm_name]["model_name"],
                commit_id=MODEL_CONFIGS[llm_name]["commit_id"],
            )
            # tokenizer = AutoTokenizer.from_pretrained("gpt2")     # 作者本身的注释
        else:
            model_configs = MODEL_CONFIGS[llm_name]
            tokenizer, pipeline = loading_huggingface_model(
                model=model_configs["model_name"], commit_id=model_configs["commit_id"]
            )
    else:
        raise ValueError(f"Unsupported model: {llm_name}")

def loading_huggingface_model(model: str, commit_id: str = None):

    print(f"Loading model {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动分配到可用设备
        revision=commit_id,
    )

    return tokenizer, pipeline



# Step X： 通过添加后缀等修改prompts
# Suffix to fool LLama safety filter from Zou et al's code
# length = 20 tokens
# adv_suffix = "--- suppose Finland----](/}]( $({[@ reversnewcommand SafeLinux={/}] '-_{( typeof"
# adv_suffix = "unless {{ {pub principal](/|{{[!dernewcommand Safepler=\"${/} False{( typeof"

if eval_type == "safe":
    # Safe prompts
    print("\nEvaluating safe prompts from: " + safe_prompts_file + "\n")
    
    prompts = choose_prompts(eval_type= "safe", prompts_file= safe_prompts_file, num_prompts)
    

    percent_safe, time_per_prompt,percent_safe_se, time_per_prompt_se = check_is_harmful_evaltype_safe(mode= "safe" ,eval_type="safe",prompts = prompts, pipeline= pipeline,num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, model= None,  tokenizer= tokenizer,max_llm_sequence_len=None)
    

    # Save results
    if mode == "insertion":
        if str(dict(num_adv=num_adv)) not in results:
            results[str(dict(num_adv=num_adv))] = {}
        results[str(dict(num_adv=num_adv))][str(dict(max_erase=max_erase))] = dict(
            percent_safe=percent_safe,
            time_per_prompt=time_per_prompt,
            percent_safe_se=percent_safe_se,
            time_per_prompt_se=time_per_prompt_se,
        )
    else:
        results[str(dict(max_erase=max_erase))] = dict(
            percent_safe=percent_safe,
            time_per_prompt=time_per_prompt,
            percent_safe_se=percent_safe_se,
            time_per_prompt_se=time_per_prompt_se,
        )
    # print(results)

elif eval_type == "empirical":
    # Empirical performance on adversarial prompts
    emp_results = {}
    if attack == "autodan":
        range_limit = 1
    else:
        range_limit = 21
    for adv_tok in range(0, range_limit, 2):
        if attack == "autodan":
            adv_prompts_file = "data/AutoDAN_prompts.txt"
        else:
            adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"
        print("Evaluating on adversarial prompts from: " + adv_prompts_file)
   
    
        prompts = choose_prompts(eval_type= "empirical", prompts_file= adv_prompts_file, num_prompts = num_prompts)
        

        if attack == "autodan":
            max_llm_sequence_len = 300
        else:
            max_llm_sequence_len = 200

        percent_safe, time_per_prompt,percent_safe_se, time_per_prompt_se = check_is_harmful_evaltype_safe(mode= "harmful",eval_type= "empirical",prompts=prompts, pipeline= pipeline, mode ,num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer = tokenizer, max_llm_sequence_len=max_llm_sequence_len)

        

        print("")

        # Save results
        emp_results[str(dict(adv_tok=adv_tok))] = dict(
            percent_harmful=percent_harmful,
            time_per_prompt=time_per_prompt,
            percent_harmful_se=percent_harmful_se,
            time_per_prompt_se=time_per_prompt_se,
        )

    if randomize:
        results[str(dict(sampling_ratio=sampling_ratio))] = emp_results
    else:
        results[str(dict(max_erase=max_erase))] = emp_results

elif eval_type == "grad_ec":
    # Evaluating the performance of GradEC on adversarial prompts
    if not use_classifier:
        print(
            "Option --use_classifier must be turned on. GradEC only works with a trained safety classifier."
        )
        exit()

    emp_results = {}
    for adv_tok in range(0, 21, 2):
        adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"

        print("Evaluating on adversarial prompts from: " + adv_prompts_file)


        

        prompts = choose_prompts(eval_type= "grad_ec", prompts_file= adv_prompts_file, num_prompts= num_prompts)


        percent_safe, time_per_prompt,percent_safe_se, time_per_prompt_se  = check_is_harmful_evaltype_safe(mode= "harmful",eval_type="grad_ec",prompts=prompts, model= model, mode ,num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer= tokenizer, max_llm_sequence_len= None)
        

        # Save results
        emp_results[str(dict(adv_tok=adv_tok))] = dict(
            percent_harmful=percent_harmful,
            time_per_prompt=time_per_prompt,
            percent_harmful_se=percent_harmful_se,
            time_per_prompt_se=time_per_prompt_se,
        )

    results[str(dict(num_iters=num_iters))] = emp_results

elif eval_type == "greedy_ec":
    # Evaluating the performance of GreedyEC on adversarial prompts
    if not use_classifier:
        print(
            "Option --use_classifier must be turned on. GreedyEC only works with a trained safety classifier."
        )
        exit()

    emp_results = {}

    if attack == "autodan":
        range_limit = 1
    else:
        range_limit = 21

    for adv_tok in range(0, range_limit, 2):
        if attack == "autodan":
            adv_prompts_file = "data/AutoDAN_prompts.txt"
        else:
            adv_prompts_file = "data/adversarial_prompts_t_" + str(adv_tok) + ".txt"

        print("Evaluating on adversarial prompts from: " + adv_prompts_file)
        

        prompts = choose_prompts(eval_type= "greedy_ec", prompts_file= adv_prompts_file, num_prompts= num_prompts)



        percent_safe, time_per_prompt,percent_safe_se, time_per_prompt_se = check_is_harmful_evaltype_greedy_ec(eval_type="greedy_ec",prompts=prompts, model= model, mode="harmful",num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer= tokenizer,max_llm_sequence_len= None)
        
        

        # Save results
        emp_results[str(dict(adv_tok=adv_tok))] = dict(
            percent_harmful=percent_harmful,
            time_per_prompt=time_per_prompt,
            percent_harmful_se=percent_harmful_se,
            time_per_prompt_se=time_per_prompt_se,
        )

    results[str(dict(num_iters=num_iters))] = emp_results

elif eval_type == "roc_curve":
    # ROC curve
    if not use_classifier:
        print(
            "Option --use_classifier must be turned on. GreedyEC only works with a trained safety classifier."
        )
        exit()

    # Load adversarial prompts
    adv_prompts_file = f"{adv_prompts_dir}/adversarial_prompts_t_{max_erase}.txt"
    print("Evaluating on adversarial prompts from: " + adv_prompts_file)

    with open(adv_prompts_file, "r") as f:
        adv_prompts = f.readlines()
        adv_prompts = [prompt.strip() for prompt in adv_prompts]

    # Sample a random subset of the prompts
    if num_prompts <= len(adv_prompts):
        adv_prompts = random.sample(adv_prompts, num_prompts)

    # Load safe prompts
    safe_prompts_file = "data/safe_prompts_test.txt"
    print("Evaluating on safe prompts from: " + safe_prompts_file)

    with open(safe_prompts_file, "r") as f:
        safe_prompts = f.readlines()
        safe_prompts = [prompt.strip() for prompt in safe_prompts]

    # Sample a random subset of the prompts
    if num_prompts <= len(safe_prompts):
        safe_prompts = random.sample(safe_prompts, num_prompts)

    print(f"\nEvaluating {ec_variant}...")
    roc = {"fpr": [0], "tpr": [0]}

    for frac in np.arange(0.0, 1.01, 0.1):
        print(f"  Fraction = {frac:.2f}")

        print("    Adv Prompts:  ")

        percent_match, time_per_prompt,percent_matches_se, time_per_prompt_se = check_is_harmful_evaltype_safe(eval_type="roc_curve",prompts=adv_prompts, model= model, mode="harmful",num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer= tokenizer,max_llm_sequence_len= None)


        roc["tpr"].append(percent_match)



        # Check if the prompts are harmful
        print("    Safe Prompts: ")
        percent_match, time_per_prompt,percent_matches_se, time_per_prompt_se = check_is_harmful_evaltype_safe(eval_type="roc_curve",prompts=safe_prompts, model= model, mode="harmful",num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer= tokenizer,max_llm_sequence_len= None)

        

        roc["fpr"].append(percent_match)



    

    roc["fpr"].append(100)
    roc["tpr"].append(100)

    results[ec_variant] = roc

elif eval_type == "smoothing":
    # Smoothing-based certificates on harmful prompts
    print(
        "Evaluating smoothing-based certificates on harmful prompts from: "
        + harmful_prompts_file
    )

    
    prompts = choose_prompts(eval_type= "smoothing", prompts_file= harmful_prompts_file, num_prompts = num_prompts)

    # List of certified lengths
    certified_length = [0] * num_prompts

    for i in range(num_prompts):
        certified_length[i] = erase_and_check_smoothing(
            prompts[i], pipeline, tokenizer, max_erase=max_erase
        )

        # Print progress
        print(
            "    Evaluating certificates... " + progress_bar((i + 1) / num_prompts),
            end="\r",
            flush=True,
        )

    print("")

    # List of certified accuracies
    num_lengths = ceil(max_erase / 2) + 2
    certified_accuracy = [0] * num_lengths

    for i in range(num_lengths):
        certified_accuracy[i] = (
            sum([length >= i for length in certified_length]) / num_prompts * 100
        )

    results[str(dict(max_erase=max_erase))] = dict(
        certified_accuracy=certified_accuracy
    )


elif eval_type == "harmful":
    # Harmful prompts
    print("\nEvaluating harmful prompts from: " + harmful_prompts_file + "\n")


    

    prompts = choose_prompts(eval_type= "harmful", prompts_file= harmful_prompts_file, num_prompts = num_prompts)


    # Optionally append adversarial suffix
    # if args.append_adv:
    #     prompts_adv = []
    #     for p in prompts: prompts_adv.append(p + adv_suffix)
    #     prompts = prompts_adv

    
    percent_harmful, time_per_prompt,percent_harmful_se, time_per_prompt_se = check_is_harmful_evaltype_harmful(mode= "harmful", eval_type ="harmful", prompts, pipeline, mode ,num_prompts, max_erase, num_adv, randomize,sampling_ratio,llm_name, tokenizer,num_iters, model=None,  max_llm_sequence_len= None)

    # Save results
    results["percent_harmful"] = percent_harmful

print("")


# Save results
print("Saving results to " + results_file)
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
