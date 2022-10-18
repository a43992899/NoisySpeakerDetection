import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_args

import torch

from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..model.model import SpeechEmbedder

NoiseLevel = Literal[0, 20, 50, 75]
NoiseType = Literal['Permute', 'Open', 'Mix']
LossType = Literal['CE', 'GE2E', 'AAM', 'AAMSC']
OptimizerType = Literal['Adam']

NOISE_LEVELS: Tuple[NoiseLevel] = get_args(NoiseLevel)
NOISE_TYPES: Tuple[NoiseType] = get_args(NoiseType)
LOSSES: Tuple[LossType] = get_args(LossType)
OPTIMIZERS: Tuple[OptimizerType] = get_args(OptimizerType)


@dataclass
class BaseConfig:
    def to_dict(self):
        return asdict(self)

    def to_json(self, json_file: Union[Path, str]):
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(**d)

    @classmethod
    def from_json(cls, json_file: Union[Path, str]):
        with open(json_file, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainConfig(BaseConfig):
    restore_model_from: str
    restore_loss_from: str

    noise_level: NoiseLevel
    noise_type: NoiseType
    loss: LossType

    N: int
    M: int
    s: int
    m: float
    K: int

    iterations: int
    optimizer: OptimizerType
    learning_rate: float

    model_lstm_hidden_size: int
    model_lstm_num_layers: int
    model_projection_size: int

    dataloader_num_workers: int
    random_seed: int

    def forge_model(self, lstm_input_size: int, num_classes: int) -> SpeechEmbedder:
        return SpeechEmbedder(
            lstm_input_size,
            self.model_lstm_hidden_size,
            self.model_lstm_num_layers,
            self.model_projection_size,
            should_softmax=(self.loss == 'CE'),
            num_classes=num_classes
        )

    def forge_criterion(self, utterance_classes_num: int):
        if self.loss == 'CE':
            self.assert_attr('N', 'M')
            return torch.nn.NLLLoss()
        elif self.loss == 'GE2E':
            self.assert_attr('N', 'M')
            return GE2ELoss()
        elif self.loss == 'AAM':
            self.assert_attr('N', 'M', 's', 'm')
            print(f'At an early stage, easy_margin is enabled for AAM. '
                  f'easy_margin flag will be turned down after {self.iterations // 8} iterations.')
            return AAMSoftmax(
                self.model_projection_size, utterance_classes_num,
                scale=self.s, margin=self.m, easy_margin=True
            )
        else:
            assert self.loss == 'AAMSC'
            self.assert_attr('N', 'M', 's', 'm', 'K')
            print(f'At an early stage, easy_margin is enabled for AAMSC. '
                  f'easy_margin flag will be turned down after {self.iterations // 8} iterations.')
            return SubcenterArcMarginProduct(
                self.model_projection_size, utterance_classes_num,
                s=self.s, m=self.m, K=self.K, easy_margin=True,
            )

    def assert_attr(self, *attrs: str):
        for attr in attrs:
            if getattr(self, attr) == -1:
                raise ValueError()

    @property
    def description(self):
        s: List[str] = []

        if self.noise_level == 0:
            s.append('clean')
        else:
            s.extend([self.noise_type, str(self.noise_level)])

        s.append(self.loss)
        s.append(f'bs{self.N * self.M}')

        if self.loss == 'GE2E':
            s.append(f'M{self.M}')
        elif self.loss == 'AAM':
            s.extend([f's{self.s}', f'm{self.m}'])
        elif self.loss == 'AAMSC':
            s.extend([f's{self.s}', f'm{self.m}', f'K{self.K}'])
        else:
            assert self.loss == 'CE'

        s.append(f'seed{self.random_seed}')

        return '-'.join(s)


@dataclass
class DataConfig(BaseConfig):
    sr: int
    nfft: int
    window: float
    hop: float
    nmels: int
    tisv_frame: int
    silence_threshold: float


@dataclass
class TestConfig(BaseConfig):
    epochs: int
    dataloader_num_workers: int

    N: int
    M: int

    random_seed: Optional[int]

    # TODO: add related training config to this model
    # model_path: str
    # iteration: str
