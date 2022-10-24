import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn.functional import cosine_similarity, normalize, one_hot, softmax, log_softmax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..constant.config import DataConfig, TrainConfig
from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..process_data.dataset import (VOX2_CLASS_NUM, SpeakerLabelDataset,
                                    SpeakerUtteranceDataset)
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory
from .beta_mixture import fit_bmm


@torch.no_grad()
def compute_distance_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
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
    label_dataset = SpeakerLabelDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
    )

    clean_memory()
    inconsistencies = np.array([], dtype=np.float32)
    noise = np.array([], dtype=np.bool8)
    for i in tqdm(range(len(label_dataset)), total=len(label_dataset), desc='Evaluating centroids and distances...'):
        utterances, is_noisy, _ = label_dataset[i]
        utterances = utterances.to(device)
        if train_config.loss == 'CE':
            embeddings: Tensor = model.get_embedding(utterances)
        else:
            embeddings: Tensor = model(utterances)
        centroid_norm = normalize(embeddings.mean(dim=0), dim=0)
        inconsistencies = np.concatenate([inconsistencies, np.fromiter(
            (
                (centroid_norm * normalize(embeddings[j, :], dim=0)).sum().item()
                for j in range(embeddings.size(0))
            ),
            dtype=np.float32
        )])
        noise = np.concatenate([noise, np.array(is_noisy)])

    return inconsistencies, noise


@torch.no_grad()
def compute_confidence_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
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
    model.load_state_dict(torch.load(
        model_dir / f'model-{selected_iteration}.pth', map_location=device,
    ))
    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    criterion = train_config.forge_criterion(utterance_classes_num).to(device).eval()
    criterion.load_state_dict(torch.load(
        model_dir / f'loss-{selected_iteration}.pth', map_location=device,
    ))

    label_dataset = SpeakerLabelDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )

    clean_memory()
    inconsistencies = []
    noise = []
    if train_config.loss == 'GE2E':
        assert isinstance(criterion, GE2ELoss)
        w = criterion.w
        b = criterion.b

        norm_centroids: List[Tensor] = []
        for i in range(len(label_dataset)):
            mel, _, label = label_dataset[i]
            assert i == label
            mel: Tensor = mel.to(device)
            centroid = normalize(model(mel).mean(dim=0), dim=0)
            norm_centroids.append(centroid)
        norm_centroids: Tensor = torch.stack(norm_centroids)

        utterance_dataset = SpeakerUtteranceDataset(
            vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
        )
        utterance_dataloader = DataLoader(utterance_dataset, batch_size=2048)
        for i in range(len(utterance_dataset)):
            # TODO use dataloader!
            mel, is_noisy, selected_id, _, _ = utterance_dataset[i]
            mel: Tensor = mel.to(device)
            norm_embedding: Tensor = normalize(model(mel), dim=0)
            all_similarities = w * (norm_centroids * norm_embedding).sum(dim=-1) + b
            y = one_hot(torch.tensor(selected_id), VOX2_CLASS_NUM)
            inconsistencies.append(torch.max(y * all_similarities).item())
            noise.append(is_noisy)

    else:
        for i in tqdm(range(len(label_dataset))):
            mel, is_noisy, label = label_dataset[i]
            mel: Tensor = mel.to(device)
            y = one_hot(torch.tensor(label), VOX2_CLASS_NUM).to(device)
            model_output: Tensor = model(mel)
            if train_config.loss in ('AAM', 'AAMSC'):
                assert isinstance(criterion, (AAMSoftmax, SubcenterArcMarginProduct))
                model_output = criterion.directly_predict(model_output)
            model_output = softmax(model_output, dim=-1)

            noise.extend(is_noisy)
            inconsistencies.extend(
                ((1 - y) * model_output[j, ...]).max().item() for j in range(model_output.size(0))
            )
    clean_memory()
    del label_dataset, utterance_dataset

    inconsistencies = np.array(inconsistencies)
    noise = np.array(noise)

    return inconsistencies, noise


@torch.no_grad()
def compute_loss_inconsistency(
    model_dir: Path, sampling_interval: int, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    if train_config.loss == 'GE2E':
        raise NotImplementedError(f'We do not evaluate loss inconsistence on GE2E.')
    data_processing_config = DataConfig.from_json(
        vox2_mel_spectrogram_dir / 'data-processing-config.json'
    )
    mislabeled_json_file = find_mislabeled_json(
        mislabeled_json_dir, train_config.noise_type, train_config.noise_level
    )
    label_dataset = SpeakerLabelDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )
    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    selected_iterations = sorted(map(
        lambda p: p.stem.split('-')[1],
        filter(
            lambda p: p.stem.startswith('model') and p.suffix == '.pth',
            model_dir.iterdir()
        )
    ), reverse=True)
    selected_iterations = selected_iterations[0:int(len(selected_iterations) * 0.8):sampling_interval]

    clean_memory()
    noisy_collected = False
    for selected_iteration in selected_iterations:
        model = train_config.forge_model(
            data_processing_config.nmels, VOX2_CLASS_NUM,
        ).to(device).eval()
        model.load_state_dict(torch.load(
            model_dir / f'model-{selected_iteration}.pth', map_location=device,
        ))
        criterion = train_config.forge_criterion(utterance_classes_num).to(device).eval()
        criterion.load_state_dict(torch.load(
            model_dir / f'loss-{selected_iteration}.pth', map_location=device,
        ))

        for i in range(len(label_dataset)):
            mel, is_noisy, label = label_dataset[i]
            mel: Tensor = mel.to(device)
            y = one_hot(torch.tensor(label), VOX2_CLASS_NUM).to(device)
            model_output = model(mel)
            if train_config.loss != 'CE':
                assert hasattr(criterion, 'directly_predict')
                model_output = criterion.directly_predict(model_output)
            assert model_output.dim() == 2
            assert model_output.size(1) == VOX2_CLASS_NUM
            model_output = log_softmax(model_output)

        noisy_collected = True
