import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, normalize, one_hot, softmax
from tqdm.auto import tqdm

from ..constant.config import DataConfig, TrainConfig
from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..process_data.dataset import (VOX2_CLASS_NUM, SpeakerLabelDataset,
                                    SpeakerUtteranceDataset)
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory, get_device


@torch.no_grad()
def compute_and_save_ge2e_embedding_centroid(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    """Pre-compute normalized embeddings centroid for GE2E."""
    device = get_device()
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    if (loss := train_config.loss) != 'GE2E':
        print(f'This routine is specially designed for GE2E loss function. Got {loss}')
        exit(1)
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

    norm_centroids: List[Tensor] = []
    for i in range(len(label_dataset)):
        mel, _, label = label_dataset[i]
        assert i == label
        mel: Tensor = mel.to(device)
        centroid = normalize(model.get_embedding(mel).mean(dim=0), dim=-1)
        norm_centroids.append(centroid)
    norm_centroids: Tensor = torch.stack(norm_centroids)

    torch.save(norm_centroids, model_dir / f'ge2e-centroids-{selected_iteration}.pth')


@torch.no_grad()
def compute_distance_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = get_device()
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
        utterances, is_noisy, label = label_dataset[i]
        assert label == i
        utterances = utterances.to(device)
        embeddings: Tensor = model.get_embedding(utterances)
        centroid_norm = normalize(embeddings.mean(dim=0), dim=0)
        inconsistencies = np.concatenate([inconsistencies, np.fromiter(
            ((centroid_norm * normalize(embeddings[j, :], dim=0)).sum().item() for j in range(embeddings.size(0))),
            dtype=np.float32
        )])
        noise = np.concatenate([noise, np.array(is_noisy)])

    return inconsistencies, noise


@torch.no_grad()
def compute_confidence_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = get_device()
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
    inconsistencies = np.array([], dtype=np.float32)
    noise = np.array([], dtype=np.float32)
    if train_config.loss == 'GE2E':
        assert isinstance(criterion, GE2ELoss)
        w = criterion.w
        b = criterion.b

        ge2e_centroid_file = model_dir / f'ge2e-centroids-{selected_iteration}.pth'
        if not ge2e_centroid_file.exists() and not ge2e_centroid_file.is_file():
            raise RuntimeError(f'Can\'t find the saved GE2E embedding centroids.')
        norm_centroids: Tensor = torch.load(ge2e_centroid_file, map_location=device)

        for i in range(len(label_dataset)):
            mels, is_noisy, selected_id = label_dataset[i]
            mels: Tensor = mels.to(device)
            norm_embedding: Tensor = normalize(model(mels), dim=-1)
            all_similarities = w * (norm_embedding @ norm_centroids.T) + b
            y = one_hot(torch.tensor(selected_id), VOX2_CLASS_NUM).to(device)
            inconsistencies = np.concatenate(
                [inconsistencies, torch.max(all_similarities * (1 - y), dim=-1)[0].detach().numpy()]
            )
            noise = np.concatenate([noise, is_noisy])
    else:
        for i in tqdm(range(len(label_dataset))):
            mels, is_noisy, label = label_dataset[i]
            mels: Tensor = mels.to(device)
            y = one_hot(torch.tensor(label), VOX2_CLASS_NUM).to(device)
            model_output: Tensor = model(mels)
            if train_config.loss in ('AAM', 'AAMSC'):
                assert isinstance(criterion, (AAMSoftmax, SubcenterArcMarginProduct))
                model_output = criterion.directly_predict(model_output)
            model_output = softmax(model_output, dim=-1)

            noise = np.concatenate([noise, is_noisy])
            inconsistencies = np.concatenate([inconsistencies, np.fromiter(
                (((1 - y) * model_output[j, ...]).max().item() for j in range(model_output.size(0))),
                dtype=np.float32
            )])
    clean_memory()

    return inconsistencies, noise


@torch.no_grad()
def compute_loss_inconsistency(
    model_dir: Path, sampling_interval: int, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = get_device()
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
    noise = np.array([], dtype=np.bool8)
    all_losses = []
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

        losses = np.array([], dtype=np.float32)
        for i in range(len(label_dataset)):
            mel, is_noisy, label = label_dataset[i]
            assert i == label
            mel: Tensor = mel.to(device)
            model_output = model(mel)
            if train_config.loss != 'CE':
                assert isinstance(criterion, (AAMSoftmax, SubcenterArcMarginProduct))
                model_output = criterion.directly_predict(model_output)
            assert model_output.dim() == 2
            assert model_output.size(1) == VOX2_CLASS_NUM
            bs = model_output.size(0)
            loss = cross_entropy(model_output, torch.tensor([label for _ in range(bs)]), reduction='none')
            losses = np.concatenate([losses, loss.detach().cpu().numpy()])
            if not noisy_collected:
                noise = np.concatenate([noise, np.array(is_noisy)])

        all_losses.append(loss)
        noisy_collected = True

    all_losses = np.stack(all_losses)
    loss_mean = all_losses.mean(axis=0)
    loss_variance = all_losses.var(axis=0)

    return noise, loss_mean, loss_variance
