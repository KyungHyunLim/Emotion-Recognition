"""
데이터셋 및 로더 관련 함수 정의

Description:
    데이터 셋을 데이터 셋 클래스 및 로더에 담아 반환

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

from typing import Dict, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from daycon_erc.utils.utils import label_dict


def load_data(
    path: str,
) -> Union[Dict, Dict, Dict, Dict]:
    """
    Description:
        데이터의 경로를 받아 불러온 후, 정해진 formatting 방법에 따라 데이터를 재구성하여 반환

    Args:
        path (str): 데이터 경로

    Returns:
        Union[Dict, Dict, Dict, Dict]
    """

    data = pd.read_csv(path)

    dialogu_id = pd.unique(data["Dialogue_ID"])
    train_id, eval_id = train_test_split(dialogu_id, test_size=0.1, random_state=42)

    train_data = data[data["Dialogue_ID"].isin(train_id)]
    eval_data = data[data["Dialogue_ID"].isin(eval_id)]

    train_data = formmating(train_data, train_id)
    eval_data = formmating(eval_data, eval_id)

    labels = pd.unique(data["Target"])
    lb_to_id, id_to_lb = label_dict(labels)

    return train_data, eval_data, lb_to_id, id_to_lb


def formmating(data: pd.DataFrame, dialogu_id: List[int]) -> Dict:
    """
    Description:
        DataFrame 형태의 데이터를 받아, formatting 후 Dictionary 형태로 반환
        Dict: {
            'sentence1': 이전 대화문 (없으면 [None])
            'sentence2': 현재 대화문
            'target': sentence2에 대한 감정
            }

    Args:
        data (pd.DataFrame): 데이터
        dialogu_id (int): 데이터 내 포함되어 있는 대화 id

    Returns:
        Dict
    """

    pre_processed_data = {"sentence1": [], "sentence2": [], "target": []}
    for d_id in dialogu_id:
        temp = data[data["Dialogue_ID"] == d_id]
        for cur, item in enumerate(
            temp.values,
        ):  # 0: ID, 1: Utterance, 2: Speaker, 3: Dialogue_ID, 4: Target

            if cur == 0:
                sentence1 = "[None]"
            else:
                sentence1 = temp.values[cur - 1][1]

            pre_processed_data["sentence1"].append(sentence1)
            pre_processed_data["sentence2"].append(item[1])
            pre_processed_data["target"].append(item[4])

    return pre_processed_data
