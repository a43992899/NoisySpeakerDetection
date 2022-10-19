import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.cuda import is_available as cuda_is_available

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerUtteranceDataset
from ..process_data.mislabel import find_mislabeled_json


def confidence_inconsistency_evaluation(
    model_dir: Path, selected_iteration: str,
    vox1_mel_spectrogram_dir: Path, vox2_mel_spectrogram_dir: Path,
    mislabeled_json_dir: Path, debug: bool,
):
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    data_processing_config = DataConfig.from_json(
        vox2_mel_spectrogram_dir / 'data-processing-config.json'
    )
    mislabeled_json_file = find_mislabeled_json(
        mislabeled_json_dir, train_config.noise_type, train_config.noise_level
    )

    model = train_config.forge_model(
        data_processing_config.nmels, VOX2_CLASS_NUM,
    ).to(device).eval()
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(model_dir / f'model-{selected_iteration}.pth')
    )
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        raise ValueError()
    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    criterion = train_config.forge_criterion(utterance_classes_num).to(device).eval()
    missing_keys, unexpected_keys = criterion.load_state_dict(
        torch.load(model_dir / f'loss-{selected_iteration}.pth')
    )
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        raise ValueError()
    
    dataset = SpeakerUtteranceDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )

    for mel, is_noisy, speaker_id, _, _ in dataset:
        embedding = model.get_embedding(mel)
    
    # TODO: do the following:
    # 1. CE do model forward, else do model get embedding and criterion predict
    # 2. call BMM and evaluation.
