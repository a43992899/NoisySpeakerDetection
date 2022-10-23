from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerLabelDataset
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory
from .beta_mixture import fit_bmm


@torch.no_grad()
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

    model = train_config.forge_model(data_processing_config.nmels, VOX2_CLASS_NUM).to(device).eval()
    model.load_state_dict(torch.load(model_dir / f'model-{selected_iteration}.pth', map_location=device))
    dataset = SpeakerLabelDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
    )

    speaker_id_data: Dict[int, Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool8]]] = dict()
    for i in range(len(dataset)):
        utterances, is_noisy, _ = dataset[i]
        embeddings: Tensor = model(utterances)
        centroid = embeddings.mean(dim=0).norm(dim=-1)
        distances = np.fromiter(
            (torch.sum(centroid * embeddings[j, ...]).item() for j in embeddings.size(0)),
            dtype=np.float32
        )
        speaker_id_data[i] = (distances, np.array(is_noisy))
