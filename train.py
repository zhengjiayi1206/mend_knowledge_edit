from transformers import HfArgumentParser

from configs import Args
from trainer import run_training


def main():
    # 解析命令行参数
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]

    # 启动训练
    run_training(args)


if __name__ == "__main__":
    main()