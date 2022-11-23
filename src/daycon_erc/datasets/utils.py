"""
데이터셋 및 로더 관련 함수 정의

Description:
    데이터 셋을 데이터 셋 클래스 및 로더에 담아 반환

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer


def tokenize_and_align_labels(
    tokenizer: AutoTokenizer,
    form: str,
    annotations: Tuple[str, List[str], str],
    max_len: int,
    label_info: Dict,
) -> Union[Dict, Dict]:
    """
    Description:
        학습을 위한 파이프라인

    Args:
        tokenizer (AutoTokenizer): 토크나이저
        form (str): 문장
        annotations (List[str, List[str], str]): 라벨 E.g ["본품#편의성", ["부직포 포장", 5, 11], "positive"]
        max_len (int): 토큰의 최대 길이
        label_info (Dict): acd, ads 태스크 라벨 id to label, label to id 딕셔너리

    Returns:
        TensorDataset
    """

    ads_count = 0
    acd_count = 0

    acd_data_dict = {"input_ids": [], "attention_mask": [], "label": []}
    ads_data_dict = {"input_ids": [], "attention_mask": [], "label": []}

    for pair in label_info["acd_pair"]:
        isPairInOpinion = False
        if pd.isna(form):
            break
        tokenized_data = tokenizer(
            form,
            pair,
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        for annotation in annotations:
            acd = annotation[0]
            ads = annotation[2]

            # # 데이터가 =로 시작하여 수식으로 인정된경우
            # if pd.isna(entity) or pd.isna(property):
            #     continue

            if ads == "------------":
                continue

            if acd == pair:
                ads_count += 1
                acd_data_dict["input_ids"].append(tokenized_data["input_ids"])
                acd_data_dict["attention_mask"].append(tokenized_data["attention_mask"])
                acd_data_dict["label"].append(label_info["acd_name_to_id"]["True"])

                ads_data_dict["input_ids"].append(tokenized_data["input_ids"])
                ads_data_dict["attention_mask"].append(tokenized_data["attention_mask"])
                ads_data_dict["label"].append(label_info["ads_name_to_id"][ads])

                isPairInOpinion = True
                break

        if isPairInOpinion is False:
            acd_count += 1
            acd_data_dict["input_ids"].append(tokenized_data["input_ids"])
            acd_data_dict["attention_mask"].append(tokenized_data["attention_mask"])
            acd_data_dict["label"].append(label_info["acd_name_to_id"]["False"])

    return acd_data_dict, ads_data_dict, ads_count, acd_count


def get_dataset(
    raw_data: List[List],
    tokenizer: AutoTokenizer,
    max_len: int,
    label_info: Dict,
) -> TensorDataset:
    """
    Description:
        학습을 위한 파이프라인

    Args:
        raw_data (List[List]): 원본 데이터
        tokenizer (AutoTokenizer): 토크나이저
        max_len (int): 토큰의 최대 길이
        label_info (Dict): acd, ads 태스크 라벨 id to label, label to id 딕셔너리

    Returns:
        TensorDataset
    """
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    ads_input_ids_list = []
    ads_attention_mask_list = []
    ads_token_labels_list = []

    global_ads_count = 0
    global_acd_count = 0

    for utterance in raw_data:
        (acd_data_dict, ads_data_dict, ads_count, acd_count) = tokenize_and_align_labels(
            tokenizer,
            utterance["sentence_form"],
            utterance["annotation"],
            max_len,
            label_info,
        )
        input_ids_list.extend(acd_data_dict["input_ids"])
        attention_mask_list.extend(acd_data_dict["attention_mask"])
        token_labels_list.extend(acd_data_dict["label"])

        ads_input_ids_list.extend(ads_data_dict["input_ids"])
        ads_attention_mask_list.extend(ads_data_dict["attention_mask"])
        ads_token_labels_list.extend(ads_data_dict["label"])

        global_ads_count += ads_count
        global_acd_count += acd_count

    print("ads_data_count: ", global_ads_count)
    print("acd_data_count: ", global_acd_count)

    return TensorDataset(
        torch.tensor(input_ids_list),
        torch.tensor(attention_mask_list),
        torch.tensor(token_labels_list),
    ), TensorDataset(
        torch.tensor(ads_input_ids_list),
        torch.tensor(ads_attention_mask_list),
        torch.tensor(ads_token_labels_list),
    )
