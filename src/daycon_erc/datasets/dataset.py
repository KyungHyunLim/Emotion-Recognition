"""
데이터셋 클래스

Description:
    torch 모델 학습시 데이터를 꺼내주기 위한 데이터셋 클래스 정의

Author:
    Name: Kyunghyun Lim
    Email: fly1294@naver.com
"""

from typing import Dict

from torch.utils.data import Dataset


class SenseDataset(Dataset):
    """데이터셋 클래스"""

    def __init__(self, args, raw_data, tokenizer, lb_to_id) -> None:
        super().__init__()

        self.data = tokenizer(
            raw_data["sentence1"],
            raw_data["sentence2"],
            max_length=args["Data"]["max_len"],
            padding="max_length",
            truncation=True,
        )

        self.labels = []
        for lb in raw_data["target"]:
            self.labels.append(lb_to_id[lb])

    def __getitem__(self, index) -> Dict:
        item = {key: val[index] for key, val in self.data.items()}
        item["labels"] = self.labels[index]
        return item

    def __len__(self) -> int:
        return len(self.labels)
