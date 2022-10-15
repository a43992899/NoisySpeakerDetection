import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Tuple, Union, get_args

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
    def from_json(cls, json_file: Union[Path, str]):
        with open(json_file, 'r') as f:
            return cls(**json.load(f))


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

    # TODO: fix iteration related stuff.
    iterations: int
    optimizer: OptimizerType
    learning_rate: float

    model_lstm_hidden_size: int
    model_lstm_num_layers: int
    model_projection_size: int

    dataloader_num_workers: int
    random_seed: int


@dataclass
class DataConfig(BaseConfig):
    sr: int
    nfft: int
    window: float
    hop: float
    nmels: int
    tisv_frame: int
    silence_threshold: float
