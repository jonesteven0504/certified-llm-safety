# bash命令中的参数配置
import argparse

def create_parser():
    """
    创建一个ArgumentParser实例并添加所有命令行参数。
    将相关的参数分组，提高可读性和可维护性。
    """
    parser = argparse.ArgumentParser(
        description="Check safety of prompts.",
        formatter_class=argparse.RawTextHelpFormatter
    )


    # 主要参数：直接创建了一个参数分组（后期可以陆续添加）
    main_group = parser.add_argument_group("主要参数")
    main_group.add_argument(
        "--num_prompts", type=int, default=2, help="要检查的提示数量"
    )
    main_group.add_argument(
        "--mode",
        type=str,
        default="suffix",
        choices=["suffix", "insertion", "infusion"],
        help="要防御的攻击模式",
    )
    main_group.add_argument(
        "--eval_type",
        type=str,
        default="safe",
        choices=[
            "safe", "harmful", "smoothing", "empirical",
            "grad_ec", "greedy_ec", "roc_curve",
        ],
        help="要评估的提示类型",
    )
    main_group.add_argument(
        "--max_erase", type=int, default=20, help="要擦除的最大 token 数量"
    )
    main_group.add_argument(
        "--num_adv",
        type=int,
        default=2,
        help="要防御的对抗性提示数量（仅限插入模式）",
    )
    main_group.add_argument(
        "--attack",
        type=str,
        default="gcg",
        choices=["gcg", "autodan"],
        help="要防御的攻击名称",
    )
    main_group.add_argument(
        "--llm_name",
        type=str,
        default="Llama-2",
        choices=["Llama-2", "Llama-2-13B", "Llama-3", "GPT-3.5"],
        help="LLM 模型名称（仅当 use_classifier=False 时使用）",
    )

    # 文件和目录路径参数
    path_group = parser.add_argument_group("文件和目录路径")
    path_group.add_argument(
        "--safe_prompts", type=str, default="data/safe_prompts.txt", help="安全提示文件路径"
    )
    path_group.add_argument(
        "--harmful_prompts", type=str, default="data/harmful_prompts.txt", help="有害提示文件路径"
    )
    path_group.add_argument(
        "--adv_prompts_dir",
        type=str,
        default="data",
        help="包含对抗性提示的目录",
    )
    path_group.add_argument(
        "--results_dir", type=str, default="results", help="保存结果的目录"
    )
    path_group.add_argument(
        "--model_wt_path",
        type=str,
        default="models/distillbert_saved_weights.pt",
        help="训练好的安全过滤器模型权重路径",
    )

    # 随机化参数
    randomize_group = parser.add_argument_group("随机化参数")
    randomize_group.add_argument(
        "--randomize", action="store_true", help="使用随机化检查"
    )
    randomize_group.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.1,
        help="评估子序列的比例（如果 randomize=True）",
    )

    # GradEC 参数
    gradec_group = parser.add_argument_group("GradEC 参数")
    gradec_group.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="GradEC 和 GreedyEC 的迭代次数",
    )
    gradec_group.add_argument(
        "--ec_variant",
        type=str,
        default="RandEC",
        choices=["RandEC", "GreedyEC", "GradEC", "GreedyGradEC"],
        help="用于 ROC 的 EC 变体",
    )

    # 其他标志参数
    misc_group = parser.add_argument_group("其他标志")
    misc_group.add_argument(
        "--use_classifier",
        action="store_true",
        help="使用自定义训练的安全过滤器",
    )
    
    return parser







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


# 设置评估文件的输出格式
def print_eval_type(eval_type:str)
    if eval_type == "safe" or eval_type == "empirical":
        print("Mode: " + mode)
        print("Maximum tokens to erase: " + str(max_erase))
        if mode == "insertion":
            print("Number of adversarial prompts to defend against: " + str(num_adv))
    elif eval_type == "smoothing" or eval_type == "roc_curve":
        print("Maximum tokens to erase: " + str(max_erase))
    elif eval_type == "grad_ec" or eval_type == "greedy_ec":
        print("Number of iterations: " + str(num_iters))
    if eval_type == "empirical" or eval_type == "grad_ec" or eval_type == "greedy_ec":
        print("Attack algorithm: " + attack)
    if eval_type == "roc_curve":
        print("EC variant: " + ec_variant)
        print("Adversarial prompts directory: " + adv_prompts_dir)
    print("* * * * * * * * * * ** * * * * * * * * * *\n", flush=True)

    return None







# start_time = time.time()
time_list = []
elapsed_time = 0



import time
def calculate_time(start_time,  elapsed_time,i ):
    current_time = time.time()  # 该参数无法直接传入，所以这里应当直接用
    time_list.append(current_time - start_time - elapsed_time)
    elapsed_time = current_time - start_time
    time_per_prompt = elapsed_time / (i + 1)

    return time_list, time_per_prompt


def erase_and_check():
    return None






TODO: 添加类型注解
TODO: 原来的函数中修改函数名称 calculate


TODO: 将原文中所有代码都进行重构、替换
def is_harmful_prompts(prompts, pipeline, tokenizer, progress_bar, ):

    # Check if the prompts are harmful
    count_safe = 0
    start_time = time.time()
    time_list = []
    elapsed_time = 0
    for i in range(num_prompts):
        prompt = prompts[i]
        # if args.append_adv: prompt += adv_suffix
        harmful = erase_and_check(
            prompt,
            pipeline,
            tokenizer,
            max_erase=max_erase,
            num_adv=num_adv,
            randomized=randomize,
            prompt_sampling_ratio=sampling_ratio,
            mode=mode,
            llm_name=llm_name,
        )

        if not harmful:
            count_safe += 1

        # current_time = time.time()
        # time_list.append(current_time - start_time - elapsed_time)
        # elapsed_time = current_time - start_time
        # time_per_prompt = elapsed_time / (i + 1)
        time_list,  time_per_prompt = calculate_time(start_time,  elapsed_time,i )

        percent_safe = count_safe / (i + 1) * 100
        print(
            "    Checking safety... "
            + progress_bar((i + 1) / num_prompts)
            + f" Detected safe = {percent_safe:5.1f}%"
            + f" Time/prompt = {time_per_prompt:5.1f}s",
            end="\r",
            flush=True,
        )
    return None