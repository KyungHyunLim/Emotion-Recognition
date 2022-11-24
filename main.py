"""
모두의 말뭉치 모델 학습을 위한 메인 파이프라인

Description:
    학습 및 추론을 위한 단계 순차 수행 - 메인 파이프라인
    1. 파라미터 로딩(args)
    2. 토크나이저, 데이터 로딩
    3. 모델 로딩
    4-1. 모델 학습(train) 및 검증(val)
    4-2. 모델 평가(test)
    5. 검증 또는 평가 결과 반환

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

import os
from argparse import ArgumentParser
from typing import Dict

import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments

from daycon_erc.datasets.dataset import SenseDataset, SenseEvalDataset
from daycon_erc.datasets.utils import load_data
from daycon_erc.logger.logger import set_logger
from daycon_erc.models.utils import get_model, get_tokenizer
from daycon_erc.utils.utils import compute_metrics, read_config, set_seed


def test(
    args: ArgumentParser,
    tokenizer: AutoTokenizer,
    label_info: Dict,
) -> Dict:
    """
    Description:
        추론 및 submission 파일 생성 및 저장

    Args:
        args (ArgumentParser): 파라미터 설정 정보
        data_reader (DataReader): 데이터를 가지고 있는 클래스
        tokenizer (AutoTokenizer): 토크나이저
        label_info (Dict): acd, ads 태스크 라벨 id to label, label to id 딕셔너리
    """
    pass


def main() -> None:
    """
    Description:
        main 함수, 학습 또는 평가 파이프라인 진행
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base.yaml")
    pars, _ = parser.parse_known_args()

    # 0. 학습 설정 로딩
    set_seed(42)
    args = read_config(pars.config)
    logger = set_logger()

    # 1. 토크나이저 불러오기
    tokenizer = get_tokenizer(args["Model"]["tokenizer"])
    special_tokens_dict = {
        "additional_special_tokens": ["[None]"],
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # 2. 모델 불러오기
    model = get_model(args["Model"]["base_model"])
    model.resize_token_embeddings(len(tokenizer))

    # 3. 데이터 불러오기
    train_data, eval_data, test_data, lb_to_id, id_to_lb = load_data(
        args["Data"]["data_path"],
        args["Data"]["test_data_path"],
    )

    if args["Flag"]["do_train"] == 0:  # 0: train // 1: test
        train_dataset = SenseDataset(args, train_data, tokenizer, lb_to_id)
        eval_dataset = SenseDataset(args, eval_data, tokenizer, lb_to_id)
        num_labels = len(id_to_lb)

        logger.info(f"start train process")
        logger.info(f"[data info] train data: {len(train_data['sentence1'])}")
        logger.info(f"[data info] eval data: {len(eval_data['sentence1'])}")
        logger.info(f"[data info] num labels: {num_labels}")

        # 4. 트레이너 셋팅
        os.environ["WANDB_PROJECT"] = args["Wandb"]["project_name"]
        trainargs = TrainingArguments(
            output_dir=args["Model"]["w_output_dir"] + args["Wandb"]["run_name"],
            overwrite_output_dir=False,
            num_train_epochs=args["Tranining"]["num_train_epochs"],
            learning_rate=args["Tranining"]["learning_rate"],
            per_device_train_batch_size=args["Tranining"]["batch_size"],
            per_device_eval_batch_size=args["Tranining"]["batch_size"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_whole macro f1 score",
            save_total_limit=3,
            fp16=True,
            run_name=args["Wandb"]["run_name"],
        )

        trainer = Trainer(
            model=model,
            args=trainargs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
    elif args["Flag"]["do_train"] == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        test_dataset = SenseEvalDataset(args, test_data, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        predictions = []
        with torch.no_grad():
            for item in track(test_dataloader, description="Inference"):
                input_ids = item["input_ids"].to(device)
                att_mask = item["attention_mask"].to(device)

                output = model(input_ids=input_ids, attention_mask=att_mask)
                output = torch.argmax(output.logits, -1)

                output = output.detach().cpu().tolist()
                predictions.extend(output)

        submission = pd.read_csv("./data/sample_submission.csv")
        predictions = [id_to_lb[item] for item in predictions]
        submission["Target"] = predictions

        path = os.path.join("./output/predictions", args["Wandb"]["run_name"])
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, "submission.csv")
        submission.to_csv(path, index=False)
    else:
        logger.info(f"wrong flag {args['Flag']['do_train']}")
        exit(2)


if __name__ == "__main__":
    main()
