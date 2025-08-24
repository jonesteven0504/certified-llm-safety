# bash命令中的参数配置
import argparse


# 该方法在多处供其他程序使用
def create_parser():
    """
    创建一个ArgumentParser实例并添加所有命令行参数。
    将相关的参数分组，提高可读性和可维护性。
    """
    parser = argparse.ArgumentParser(
        description="Check safety of prompts and Adversarial masks for the safety classifier.",
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
    





    # 训练参数组
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument(
        "--safe_train",
        type=str,
        default="data/safe_prompts_train_insertion_erased.txt",
        help="安全提示训练数据文件的路径"
    )

    train_group.add_argument(
        "--harmful_train",
        type=str,
        default="data/harmful_prompts_train.txt",
        help="有害提示训练数据文件的路径")

    train_group.add_argument(
        "--safe_test",
        type=str,
        default="data/safe_prompts_test_insertion_erased.txt",
        help="安全提示测试数据文件的路径"
    )

    train_group.add_argument(
        "--harmful_test",
        type=str,
        default="data/harmful_prompts_test.txt",
        help="有害提示测试数据文件的路径"
    )

    train_group.add_argument(
        "--save_path",
        type=str,
        default="models/distilbert_insertion.pt",
        help="模型保存路径"
    )

    # 训练相关的参数
    train_group.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )
    train_group.add_argument(
        "--loss_fn",
        type=str,
        default="nn.CrossEntropyLoss()",
        help="直接指定损失函数，如: nn.CrossEntropyLoss(), nn.MSELoss(), nn.NLLLoss(weight=weights)"
    )
    return parser


