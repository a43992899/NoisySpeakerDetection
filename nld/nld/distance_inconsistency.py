from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.cuda import is_available as cuda_is_available

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerUtteranceDataset
from ..process_data.mislabel import find_mislabeled_json


def distance_inconsistency_evaluation(
    model_dir: Path, selected_iteration: str,
    vox1_mel_spectrogram_dir: Path, vox2_mel_spectrogram_dir: Path,
    mislabeled_json_dir: Path, debug: bool,
):
    # TODO: log to wandb?
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    data_processing_config = DataConfig.from_json(
        vox2_mel_spectrogram_dir / 'data-processing-config.json'
    )
    mislabeled_json_file = find_mislabeled_json(
        mislabeled_json_dir, train_config.noise_type, train_config.noise_level
    )

    model = train_config.forge_model(
        data_processing_config.nmels, VOX2_CLASS_NUM
    ).to(device).eval()
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(model_dir / f'model-{selected_iteration}.pth')
    )
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        raise ValueError()
    dataset = SpeakerUtteranceDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )

    speaker_utterances: Dict[int, List[Tuple[Tensor, bool]]] = dict()

    for mel, is_noisy, speaker_id, _, _ in dataset:
        embedding = model.get_embedding(mel)
        try:
            speaker_utterances[speaker_id].append((embedding, is_noisy))
        except KeyError:
            speaker_utterances[speaker_id] = (embedding, is_noisy)

    speaker_centroids = {
        k: torch.stack([t for t, _ in v]).mean(dim=0)
        for k, v in speaker_utterances.items()
    }

    speaker_utterances_distance = {
        k: [(torch.sqrt(torch.sum(
            (t - speaker_centroids[k]) ** 2
        )), is_noisy) for t, is_noisy in v]
        for k, v in speaker_utterances.items()
    }

    # TODO: how to call BMM?
