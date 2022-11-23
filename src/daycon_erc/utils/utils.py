"""
모델 학습을 각종 유틸함수

Description:
    설정 딕셔너리 로딩 등의 유틸함수 모음

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

from typing import Dict

import yaml


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
