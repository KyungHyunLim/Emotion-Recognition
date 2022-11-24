"""
모델 학습을 각종 유틸함수

Description:
    설정 딕셔너리 로딩 등의 유틸함수 모음

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

import random
from typing import Dict, List, Union

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from torch.backends import cudnn



def read_config(path: str) -> Dict:
    """
    Description:
        학습을 설정 yaml 파일 로딩

    Args:
        path (str): yaml 파일 로딩
    """
    with open(path, encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def label_dict(labels: List[str]) -> Union[Dict, Dict]:
    """
    Description:
        라벨변환을 위한 dictionary 생성

    Args:
        labels (List[str]): 라벨 종류
    """

    lb_to_id = {}
    id_to_lb = {}
    for idx, lb in enumerate(sorted(labels)):
        lb_to_id[lb] = idx
        id_to_lb[idx] = lb

    return lb_to_id, id_to_lb


def set_seed(seed: int) -> None:
    """
    Description:
        랜덤 시드를 고정

    Args:
        seed (int): 랜덤 시드고정을 위한 시드값
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def compute_metrics(pred: List[torch.Tensor]) -> Dict:
    """
    Description:
        라벨 별 및 전체 metric 계산

    Args:
        pred (List[torch.Tensor]): target 및 예측값
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    kind_of_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    # calculate accuracy using sklearn's function
    result = classification_report(labels, preds, output_dict=True, target_names=kind_of_labels)

    result_dict = {}
    result_dict["whole macro f1 score"] = result["macro avg"]["f1-score"]
    result_dict["acc"] = result["accuracy"]

    for lb in kind_of_labels:
        result_dict[f"{lb} f1 score"] = result[lb]["f1-score"]

    return result_dict
