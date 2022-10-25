from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn.functional import normalize
from tqdm.auto import tqdm

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerLabelDataset
from ..process_data.mislabel import find_mislabeled_json
from ..utils import get_device
from .inconsistence import (compute_confidence_inconsistency,
                            compute_distance_inconsistency,
                            compute_loss_inconsistency)


@torch.no_grad()
def compute_and_save_ge2e_embedding_centroid(
    all_model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    """Pre-compute normalized embeddings centroid for GE2E."""
    device = get_device()
    data_processing_config = DataConfig.from_json(
        vox2_mel_spectrogram_dir / 'data-processing-config.json'
    )
    model = train_config.forge_model(data_processing_config.nmels, VOX2_CLASS_NUM).to(device).eval()

    noise_dict: Dict[Any, List[Path]] = {
        ('Permute', 20): [],
        ('Permute', 50): [],
        ('Permute', 75): [],
        ('Open', 20): [],
        ('Open', 50): [],
        ('Open', 75): [],
        0: []
    }
    for model_dir in all_model_dir.iterdir():
        if not model_dir.is_dir() or 'seed0' not in model_dir.name or 'GE2E' not in model_dir.name:
            continue
        train_config = TrainConfig.from_json(model_dir / 'config.json')
        assert train_config.loss == 'GE2E'
        if train_config.noise_level == 0:
            noise_dict[0].append(model_dir)
        if train_config.noise_type == 'Permute' and train_config.noise_level == 20:
            noise_dict[('Permute', 20)].append(model_dir)
        if train_config.noise_type == 'Permute' and train_config.noise_level == 50:
            noise_dict[('Permute', 50)].append(model_dir)
        if train_config.noise_type == 'Permute' and train_config.noise_level == 75:
            noise_dict[('Permute', 75)].append(model_dir)
        if train_config.noise_type == 'Open' and train_config.noise_level == 20:
            noise_dict[('Open', 20)].append(model_dir)
        if train_config.noise_type == 'Open' and train_config.noise_level == 50:
            noise_dict[('Open', 50)].append(model_dir)
        if train_config.noise_type == 'Open' and train_config.noise_level == 75:
            noise_dict[('Open', 75)].append(model_dir)

    for key, model_dirs in noise_dict.items():
        train_config = TrainConfig.from_json(model_dirs[0] / 'config.json')
        mislabeled_json_file = find_mislabeled_json(
            mislabeled_json_dir, train_config.noise_type, train_config.noise_level
        )
        label_dataset = SpeakerLabelDataset(
            vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
        )

        tensors: Dict[Path, List[Tensor]] = dict()
        for i in tqdm(range(len(label_dataset)), total=len(label_dataset), desc=str(key)):
            mel, _, label = label_dataset[j]
            assert i == label
            mel: Tensor = mel.to(device)
            for j, model_dir in enumerate(model_dirs):
                model.load_state_dict(
                    torch.load(all_model_dir / f'model-{selected_iteration}.pth', map_location=device)
                )
                centroid = normalize(model.get_embedding(mel).mean(dim=0), dim=-1)
                try:
                    tensors[model_dir].append(centroid)
                except LookupError:
                    tensors[model_dir] = [centroid]

        for model_dir, centroids in tensors.items():
            centroids = torch.stack(centroids)
            torch.save(centroids, model_dir / f'ge2e-centroids-{selected_iteration}.pth')


def nld_distance_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: List[str] = args.selected_iterations
    if selected_iterations is None:
        selected_iterations = ['final']
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    for iteration in selected_iterations:
        compute_distance_inconsistency(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )


def nld_confidence_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: List[str] = args.selected_iterations
    if selected_iterations is None:
        selected_iterations = ['final']
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    for iteration in selected_iterations:
        compute_confidence_inconsistency(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )


def nld_loss_main(args):
    model_dir: Path = args.model_dir
    sampling_interval: int = args.sampling_interval
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    compute_loss_inconsistency(
        model_dir, sampling_interval, vox1_mel_spectrogram_dir,
        vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
    )
