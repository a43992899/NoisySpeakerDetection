import json
from pathlib import Path

import wandb
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn.functional import normalize, one_hot, softmax
from tqdm.auto import tqdm

from ..constant.entities import WANDB_NLD_PROJECT_NAME, WANDB_ENTITY
from ..constant.config import DataConfig, NoiseLevel, NoisyLabelDetectionConfig, TrainConfig
from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerDataset
from ..process_data.mislabel import find_mislabeled_json
from ..utils import clean_memory, get_device
from .beta_mixture import fit_bmm


def compute_precision(ypreds: npt.NDArray, ylabels: npt.NDArray, noise_level: NoiseLevel):
    selected_indices = np.argsort(ypreds)[-int(len(ypreds) * noise_level / 100):]
    selected_ylabels = ylabels[selected_indices]
    return selected_ylabels.sum() / len(selected_ylabels)


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

    nld_config = NoisyLabelDetectionConfig(train_config, 'Distance')

    model = train_config.forge_model(data_processing_config.nmels, VOX2_CLASS_NUM).to(device).eval()
    model.load_state_dict(
        torch.load(model_dir / f'model-{selected_iteration}.pth', map_location=device)
    )
    dataset = SpeakerDataset(
        -1, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )

    if not debug:
        wandb.init(
            project=WANDB_NLD_PROJECT_NAME,
            entity=WANDB_ENTITY,
            name=f'distance-{train_config.description}',
            config=nld_config.to_dict()
        )

    clean_memory()
    inconsistencies = np.array([], dtype=np.float32)
    is_noises = np.array([], dtype=np.bool8)
    for i in tqdm(range(len(dataset)), total=len(dataset), desc='Evaluating centroids and distances...'):
        mels, is_noisy, labels, _, _ = dataset[i]
        assert torch.all(i == labels).item() is True
        mels = mels.to(device)

        batch_size = mels.size(0)
        mel_chunks = [mels[m:m + 512, ...] for m in range(0, 512 * ((batch_size - 1) // 512 + 1), 512)]
        embeddings = [model.get_embedding(mel_chunk) for mel_chunk in mel_chunks]
        embeddings = torch.cat(embeddings, dim=0)
        assert embeddings.size(0) == batch_size

        centroid_norm = normalize(embeddings.mean(dim=0), dim=-1)
        embeddings_norm = normalize(embeddings, dim=-1)
        inconsistencies = np.concatenate([
            inconsistencies,
            1 - (embeddings_norm @ centroid_norm.unsqueeze(-1)).flatten().cpu().numpy()
        ], dtype=np.float32)
        is_noises = np.concatenate([is_noises, np.array(is_noisy)], dtype=np.bool8)
        if i % 100 == 0:
            precision = compute_precision(inconsistencies, is_noises, train_config.noise_level)
            if debug:
                print(f'{i = }, {precision = :.4f}')
            else:
                wandb.log({'Intermediate Precision': precision})

    precision = compute_precision(inconsistencies, is_noises, train_config.noise_level)

    bmm_model, _, _ = fit_bmm(inconsistencies, max_iters=10, rm_outliers=True)
    estimated_noise_level = bmm_model.weight[1]

    if debug:
        print(f'{bmm_model.weight = }')
        print(f'{estimated_noise_level = }')
        print(f'{precision = }')
    else:
        wandb.log({
            'Estimated Noise Level': estimated_noise_level,
            'Precision': precision
        })
        wandb.finish()
        np.save(model_dir / f'nld-distance-inconsistencies-{selected_iteration}.npy', inconsistencies)
        np.save(model_dir / f'nld-distance-noise-labels-{selected_iteration}.npy', is_noises)


@torch.no_grad()
def compute_confidence_inconsistency(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = get_device()
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    data_processing_config = DataConfig.from_json(vox2_mel_spectrogram_dir / 'data-processing-config.json')
    mislabeled_json_file = find_mislabeled_json(mislabeled_json_dir, train_config.noise_type, train_config.noise_level)

    nld_config = NoisyLabelDetectionConfig(train_config, 'Confidence')

    model = train_config.forge_model(data_processing_config.nmels, VOX2_CLASS_NUM,).to(device).eval()
    model.load_state_dict(torch.load(model_dir / f'model-{selected_iteration}.pth', map_location=device,))
    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    criterion = train_config.forge_criterion(utterance_classes_num).to(device).eval()
    criterion.load_state_dict(torch.load(model_dir / f'loss-{selected_iteration}.pth', map_location=device))
    criterion.eval()

    if not debug:
        wandb.init(
            project=WANDB_NLD_PROJECT_NAME,
            entity=WANDB_ENTITY,
            name=f'confidence-{train_config.description}',
            config=nld_config.to_dict()
        )

    dataset = SpeakerDataset(-1, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file)

    clean_memory()
    inconsistencies = np.array([], dtype=np.float32)
    is_noises = np.array([], dtype=np.bool8)
    if train_config.loss == 'GE2E':
        assert isinstance(criterion, GE2ELoss)
        w = criterion.w
        b = criterion.b

        ge2e_centroid_file = model_dir / f'ge2e-centroids-{selected_iteration}.pth'
        if not ge2e_centroid_file.exists() and not ge2e_centroid_file.is_file():
            raise RuntimeError(f'Can\'t find the saved GE2E embedding centroids.')
        norm_centroids: Tensor = torch.load(ge2e_centroid_file, map_location=device)

        for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Processing {model_dir.stem}'):
            mels, is_noisy, labels, _, _ = dataset[i]
            assert torch.all(i == labels).item() is True
            mels = mels.to(device)

            batch_size = mels.size(0)
            mel_chunks = [mels[m:m + 512, ...] for m in range(0, 512 * ((batch_size - 1) // 512 + 1), 512)]
            embeddings = [model.get_embedding(mel_chunk) for mel_chunk in mel_chunks]
            embeddings = torch.cat(embeddings, dim=0)
            assert embeddings.size(0) == batch_size

            norm_embedding: Tensor = normalize(embeddings, dim=-1)
            all_similarities = w * (norm_embedding @ norm_centroids.T) + b
            all_similarities = softmax(all_similarities, dim=-1)
            y = one_hot(torch.tensor(labels), VOX2_CLASS_NUM).to(device)
            inconsistencies = np.concatenate([
                inconsistencies, torch.min(1 - all_similarities * y, dim=-1)[0].detach().cpu().numpy()
            ], dtype=np.float32)
            is_noises = np.concatenate([is_noises, is_noisy], dtype=np.bool8)
            if i % 100 == 0:
                precision = compute_precision(inconsistencies, is_noises, train_config.noise_level)
                if debug:
                    print(f'{i = }, {precision = :.4f}')
                else:
                    wandb.log({'Intermediate Precision': precision})
    else:
        for i in tqdm(range(len(dataset)), total=len(dataset), desc=f'Processing {model_dir.stem}'):
            mels, is_noisy, labels, _, _ = dataset[i]
            assert torch.all(i == labels).item() is True
            mels: Tensor = mels.to(device)
            y = one_hot(torch.tensor(labels), VOX2_CLASS_NUM).to(device)

            batch_size = mels.size(0)
            mel_chunks = [mels[m:m + 512, ...] for m in range(0, 512 * ((batch_size - 1) // 512 + 1), 512)]
            model_output = [model(mel_chunk, logsoftmax=False) for mel_chunk in mel_chunks]
            model_output = torch.cat(model_output, dim=0)
            assert model_output.size(0) == batch_size

            if train_config.loss in ('AAM', 'AAMSC'):
                assert isinstance(criterion, (AAMSoftmax, SubcenterArcMarginProduct))
                model_output = criterion.directly_predict(model_output)
                model_output = softmax(model_output, dim=-1)
            inconsistencies = np.concatenate([
                inconsistencies, (1 - y * model_output).min(dim=-1)[0].detach().cpu().numpy()
            ], dtype=np.float32)
            is_noises = np.concatenate([is_noises, is_noisy], dtype=np.bool8)
            if i % 100 == 0:
                precision = compute_precision(inconsistencies, is_noises, train_config.noise_level)
                if debug:
                    print(f'{i = }, {precision = :.4f}')
                else:
                    wandb.log({'Intermediate Precision': precision})

    precision = compute_precision(inconsistencies, is_noises, train_config.noise_level)

    bmm_model, _, _ = fit_bmm(inconsistencies, max_iters=10, rm_outliers=True)
    estimated_noise_level = bmm_model.weight[1]

    if debug:
        print(f'{bmm_model.weight = }')
        print(f'{estimated_noise_level = }')
        print(f'{precision = }')
    else:
        wandb.log({
            'Estimated Noise Level': estimated_noise_level,
            'Precision': precision
        })
        wandb.finish()
        np.save(model_dir / f'nld-confidence-inconsistencies-{selected_iteration}.npy', inconsistencies)
        np.save(model_dir / f'nld-confidence-noise-labels-{selected_iteration}.npy', is_noises)
