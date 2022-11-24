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

from transformers import AutoTokenizer

from daycon_erc.datasets.utils import load_data
from daycon_erc.logger.logger import set_logger
from daycon_erc.models.utils import get_tokenizer
from daycon_erc.utils.utils import read_config, set_seed


def train(
    args: ArgumentParser,
    tokenizer: AutoTokenizer,
    label_info: Dict,
) -> None:
    """
    Description:
        학습을 위한 파이프라인

    Args:
        args (ArgumentParser): 파라미터 설정 정보
        data_reader (DataReader): 데이터를 가지고 있는 클래스
        tokenizer (AutoTokenizer): 토크나이저
        label_info (Dict): acd, ads 태스크 라벨 id to label, label to id 딕셔너리
    """
    pass


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

    # 2. 데이터 불러오기
    train_data, eval_data, lb_to_id, id_to_lb = load_data(args["Data"]["data_path"])
    num_labels = len(lb_to_id)

    print(num_labels)

    # if args.do_train:
    #     train(args, data_reader, tokenizer, label_info)
    # elif args.do_test:
    #     test(args, data_reader, tokenizer, label_info)


if __name__ == "__main__":
    main()
