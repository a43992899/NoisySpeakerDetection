from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerDataset
from ..process_data.mislabel import find_mislabeled_json
from ..utils import get_device


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

    noise_dict: Dict[Any, List[Path]] = {
        0: [],
        ('Permute', 20): [],
        ('Permute', 50): [],
        ('Permute', 75): [],
        ('Open', 20): [],
        ('Open', 50): [],
        ('Open', 75): [],
    }
    for model_dir in all_model_dir.iterdir():
        if not model_dir.is_dir() or 'seed0' not in model_dir.name or 'GE2E' not in model_dir.name:
            continue
        train_config = TrainConfig.from_json(model_dir / 'config.json')
        assert train_config.loss == 'GE2E'
        if train_config.noise_level == 0:
            noise_dict[0].append(model_dir)
        elif train_config.noise_type == 'Permute' and train_config.noise_level == 20:
            noise_dict[('Permute', 20)].append(model_dir)
        elif train_config.noise_type == 'Permute' and train_config.noise_level == 50:
            noise_dict[('Permute', 50)].append(model_dir)
        elif train_config.noise_type == 'Permute' and train_config.noise_level == 75:
            noise_dict[('Permute', 75)].append(model_dir)
        elif train_config.noise_type == 'Open' and train_config.noise_level == 20:
            noise_dict[('Open', 20)].append(model_dir)
        elif train_config.noise_type == 'Open' and train_config.noise_level == 50:
            noise_dict[('Open', 50)].append(model_dir)
        else:
            assert train_config.noise_type == 'Open' and train_config.noise_level == 75
            noise_dict[('Open', 75)].append(model_dir)

    for key, model_dirs in noise_dict.items():
        train_config = TrainConfig.from_json(model_dirs[0] / 'config.json')
        mislabeled_json_file = find_mislabeled_json(
            mislabeled_json_dir, train_config.noise_type, train_config.noise_level,
        )
        dataset = SpeakerDataset(
            -1, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
        )
        model = train_config.forge_model(
            data_processing_config.nmels, VOX2_CLASS_NUM
        ).to(device).eval()

        tensors: Dict[Path, List[Tensor]] = dict()
        for i in tqdm(range(len(dataset)), total=len(dataset), desc=str(key)):
            mels, _, labels, _, _ = dataset[i]
            assert torch.all(i == labels).item() is True
            mels: Tensor = mels.to(device)
            for model_dir in model_dirs:
                model.load_state_dict(
                    torch.load(model_dir / f'model-{selected_iteration}.pth', map_location=device)
                )

                batch_size = mels.size(0)
                mel_chunks = [mels[m:m + 512, ...] for m in range(0, 512 * (batch_size // 512 + 1), 512)]

                embeddings = []
                for mel_chunk in mel_chunks:
                    embeddings_chunk = model.get_embedding(mel_chunk)
                    embeddings.append(embeddings_chunk)
                embeddings = torch.cat(embeddings, dim=0)

                centroid = embeddings.mean(dim=0)
                centroid_norm = torch.clone(normalize(centroid, dim=-1)).cpu()
                try:
                    tensors[model_dir].append(centroid_norm)
                except LookupError:
                    tensors[model_dir] = [centroid_norm]
            if debug and i == 200:
                break

        for model_dir, centroids in tensors.items():
            centroids = torch.stack(centroids)
            if not debug:
                torch.save(centroids, model_dir / f'ge2e-centroids-{selected_iteration}.pth')
