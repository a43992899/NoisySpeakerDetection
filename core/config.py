# -*- coding: utf-8 -*-
#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import yaml


@dataclass
class DataConfig:
    train_path: str
    test_path: str
    train_path: str
    test_path: str
    nld_path: str
    vox1_path: str
    esc50_path: str
    data_preprocessed: bool
    sr: int
    nfft: int
    window: float
    hop: float
    nmels: int
    tisv_frame: int


@dataclass
class ModelConfig:
    hidden: int
    num_layer: int
    proj: int


@dataclass
class TrainConfig:
    debug: bool
    restore: bool
    noise_type: Literal['Permute', 'Open', 'Mix']
    noise_level: Literal[0, 20, 50, 75]
    N: int
    M: int
    num_workers: int
    lr: float
    optimizer: str
    loss: Literal['CE', 'GE2E', 'AAM', 'AAMSC']
    s: int
    m: float
    K: int
    epochs: int
    log_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    model_path: str


@dataclass
class TestConfig:
    N: int
    M: int
    num_workers: int
    epochs: int
    model_path: str


@dataclass
class NLDConfig:
    noise_type: Literal['Permute', 'Open', 'Mix']
    noise_level: Literal[0, 20, 50, 75]
    model_path: str
    N: int
    M: int
    num_workers: int


@dataclass
class Config:
    stage: Literal['train', 'test', 'nld']
    device: Literal['cpu', 'cuda']
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    test: TestConfig
    nld: NLDConfig

    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.device = kwargs['device']
        self.data = DataConfig(**kwargs['data'])
        self.model = ModelConfig(**kwargs['model'])
        self.train = TrainConfig(**kwargs['train'])
        self.test = TestConfig(**kwargs['test'])
        self.nld = NLDConfig(**kwargs['nld'])

    @classmethod
    def from_yaml(cls, file: Union[Path, str]):
        with open(file, 'r') as stream:
            docs = yaml.load_all(stream, Loader=yaml.FullLoader)
            hp = dict()
            for doc in docs:
                for k, v in doc.items():
                    hp[k] = v
        return Config(**hp)


Hparam = Config.from_yaml

MIX_BS128_CONFIG_PATHS = '''
data/models/Mix/Softmax/20%/bs128/log/config.yaml
data/models/Mix/Softmax/50%/bs128/log/config.yaml
data/models/Mix/Softmax/75%/bs128/log/config.yaml
data/models/Mix/GE2E/20%/m8_bs128/log/config.yaml
data/models/Mix/GE2E/50%/m8_bs128/log/config.yaml
data/models/Mix/GE2E/75%/m16_bs128/log/config.yaml
data/models/Mix/AAMSC/20%/m0.1_s15_k10_bs128/log/config.yaml
data/models/Mix/AAMSC/50%/m0.1_s15_k10_bs128/log/config.yaml
data/models/Mix/AAMSC/75%/m0.1_s15_k10_bs128/log/config.yaml
data/models/Mix/AAM/20%/m0.1_s15_bs128/log/config.yaml
data/models/Mix/AAM/50%/m0.1_s15_bs128/log/config.yaml
data/models/Mix/AAM/75%/m0.1_s15_bs128/log/config.yaml
'''

AAMSC_BS128_CONFIG_PATHS = '''
data/models/Open/AAMSC/20%/m0.1_s15_k3_bs128/log/config.yaml
data/models/Open/AAMSC/50%/m0.1_s15_k3_bs128/log/config.yaml
data/models/Open/AAMSC/75%/m0.1_s15_k3_bs128/log/config.yaml
data/models/Mix/AAMSC/20%/m0.1_s15_k3_bs128/log/config.yaml
data/models/Mix/AAMSC/50%/m0.1_s15_k3_bs128/log/config.yaml
data/models/Mix/AAMSC/75%/m0.1_s15_k3_bs128/log/config.yaml
'''

MIX_BS256_CONFIG_PATHS = '''
data/models/Mix/Softmax/20%/bs256/log/config.yaml
data/models/Mix/Softmax/50%/bs256/log/config.yaml
data/models/Mix/Softmax/75%/bs256/log/config.yaml
data/models/Mix/GE2E/20%/m8_bs256/log/config.yaml
data/models/Mix/GE2E/50%/m8_bs256/log/config.yaml
data/models/Mix/GE2E/75%/m16_bs256/log/config.yaml
data/models/Mix/AAMSC/20%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Mix/AAMSC/50%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Mix/AAMSC/75%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Mix/AAM/20%/m0.1_s15_bs256/log/config.yaml
data/models/Mix/AAM/50%/m0.1_s15_bs256/log/config.yaml
data/models/Mix/AAM/75%/m0.1_s15_bs256/log/config.yaml
'''

OPEN_BS256_CONFIG_PATHS = '''
data/models/Open/Softmax/20%/bs256/log/config.yaml
data/models/Open/Softmax/50%/bs256/log/config.yaml
data/models/Open/Softmax/75%/bs256/log/config.yaml
data/models/Open/GE2E/20%/m8_bs256/log/config.yaml
data/models/Open/GE2E/50%/m8_bs256/log/config.yaml
data/models/Open/GE2E/75%/m16_bs256/log/config.yaml
data/models/Open/AAMSC/20%/m0.1_s15_k10_bs256/log/confi
data/models/Open/AAMSC/50%/m0.1_s15_k10_bs256/log/confi
data/models/Open/AAMSC/75%/m0.1_s15_k10_bs256/log/confi
data/models/Open/AAM/20%/m0.1_s15_bs256/log/config.yaml
data/models/Open/AAM/50%/m0.1_s15_bs256/log/config.yaml
data/models/Open/AAM/75%/m0.1_s15_bs256/log/config.yaml
'''

PERMUTE_BS256_CONFIG_PATHS = '''
data/models/Permute/Softmax/20%/bs256/log/config.yaml
data/models/Permute/Softmax/50%/bs256/log/config.yaml
data/models/Permute/Softmax/75%/bs256/log/config.yaml
data/models/Permute/GE2E/20%/m8_bs256/log/config.yaml
data/models/Permute/GE2E/50%/m8_bs256/log/config.yaml
data/models/Permute/GE2E/75%/m16_bs256/log/config.yaml
data/models/Permute/AAMSC/20%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Permute/AAMSC/50%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Permute/AAMSC/75%/m0.1_s15_k10_bs256/log/config.yaml
data/models/Permute/AAM/20%/m0.1_s15_bs256/log/config.yaml
data/models/Permute/AAM/50%/m0.1_s15_bs256/log/config.yaml
data/models/Permute/AAM/75%/m0.1_s15_bs256/log/config.yaml
'''

REMAINING_CONFIG_PATHS = '''
data/models/Open/AAMSC/20%/m0.1_s15_k3_bs256/log/config.yaml
data/models/Open/AAMSC/50%/m0.1_s15_k3_bs256/log/config.yaml
data/models/Open/AAMSC/75%/m0.1_s15_k3_bs256/log/config.yaml
'''
