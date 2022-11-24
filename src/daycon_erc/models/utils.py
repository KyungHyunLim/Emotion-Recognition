"""
모델 및 토크나이저를 불러오기 위한 유틸

Description:
    src.modu_corpus.models에 정의되어있는 토크나이저 및 모델을
    간단하게 불러오기 위한 유틸리티 함수 정의

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""


from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Description:
        토크나이저의 이름을 이용해 반환
        Hugging face 토크나이저의 경우, AutoTokenizer를 활용
        직접 학습한 Tokenizer의 경우, 새로 정의해서 따로 로딩 (틀을 허깅페이스에 맞추면 좋음)

    Args:
        tokenizer_name (str): 토크나이저 모델 이름 (허깅페이스용)

    Returns:
        AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def get_model(model_name: str) -> AutoModelForSequenceClassification:
    """
    Description:
        모델의 이름을 이용해 모델 반환
        Hugging face 모델의 경우, AutoModelForSequenceClassification 또는 태스크에 맞는 클래스 활용

        직접 커스텀한 모델의 경우, 경로를 이용해서 로딩 (틀을 허깅페이스에 맞추면 좋음)

    Args:
        model_name (str): 모델 이름 (허깅페이스용)

    Returns:
        AutoModelForSequenceClassification
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
    return model
