import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, normalize, one_hot, softmax, cosine_similarity
from tqdm.auto import tqdm

from nld.nld.beta_mixture import fit_bmm

from ..constant.config import DataConfig, TrainConfig
from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..process_data.dataset import (VOX2_CLASS_NUM, SpeakerDataset)
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory, get_device


def compute_precision(ypreds: npt.NDArray, ylabels: npt.NDArray, noise_level: npt.NDArray):
    selected_indices = np.argsort(ypreds)[-int(len(ypreds) * noise_level / 100):]
    selected_ypreds = ypreds[selected_indices]
    selected_ylabels = ylabels[selected_indices]
    # compute precision
    return selected_ylabels.sum() / len(selected_ylabels)


@torch.no_grad()
def compute_distance_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    print(f'{model_dir = }')
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
    dataset = SpeakerDataset(
        -1, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file,
    )

    clean_memory()
    inconsistencies = np.array([], dtype=np.float32)
    is_noisies = np.array([], dtype=np.bool8)
    for i in tqdm(range(len(dataset)), total=len(dataset), desc='Evaluating centroids and distances...'):
        utterances, is_noisy, label, corrupted_files, _ = dataset[i]
        utterances = utterances.to(device)
        embeddings: Tensor = model.get_embedding(utterances)
        centroid_norm = normalize(embeddings.mean(dim=0), dim=-1)
        embeddings_norm = normalize(embeddings, dim=-1)
        inconsistencies = np.concatenate([
            inconsistencies,
            1 - (embeddings_norm @ centroid_norm.unsqueeze(-1)).flatten().cpu().numpy()
        ])
        is_noisies = np.concatenate([is_noisies, np.array(is_noisy)])
        if i == 20:
            break

    precision = compute_precision(inconsistencies, is_noisies, train_config.noise_level)

    bmm_model, bmm_model_max, bmm_model_min = fit_bmm(inconsistencies, max_iters=50, rm_outliers=True)
    if train_config.noise_level >= 70:
        estimated_noise_level = bmm_model.weight[0]
    else:
        estimated_noise_level = bmm_model.weight[1]
    print(f'{bmm_model.weight = }')
    print(f'{precision = }')
    print(f'{estimated_noise_level = }')

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
