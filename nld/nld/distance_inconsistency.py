from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerUtteranceDataset
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory
from .beta_mixture import fit_bmm


def distance_inconsistency_evaluation(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
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
        data_processing_config.nmels, VOX2_CLASS_NUM,
    ).to(device).eval()
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(model_dir / f'model-{selected_iteration}.pth')
    )
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        raise ValueError()
    dataset = SpeakerUtteranceDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
    )

    speaker_utterances: Dict[int, List[Tensor, bool]] = dict()
    speaker_utterance_is_noisy: Dict[int, List[bool]] = dict()
    for i in len(dataset):
        mel, is_noisy, speaker_id, _, _ = dataset[i]
        normalized_embedding = model.get_embedding(mel.to(device)).norm()
        try:
            speaker_utterances[speaker_id].append(normalized_embedding)
        except KeyError:
            speaker_utterances[speaker_id] = [normalized_embedding]
        try:
            speaker_utterance_is_noisy[speaker_id].append(is_noisy)
        except KeyError:
            speaker_utterance_is_noisy[speaker_id] = [is_noisy]

    speaker_centroids = {
        k: torch.stack([t for t in v]).mean(dim=0).norm()
        for k, v in speaker_utterances.items()
    }
    speaker_utterances_distance: Dict[int, npt.NDArray] = {
        k: np.fromiter((cosine_similarity(t, speaker_centroids[k]).item() for t in v), dtype=np.float32)
        for k, v in speaker_utterances.items()
    }

    del speaker_utterances, speaker_centroids
    clean_memory()

    for speaker_id in speaker_utterances_distance.keys():
        distances = speaker_utterances_distance[speaker_id]
        is_noisy = speaker_utterance_is_noisy[speaker_id]
