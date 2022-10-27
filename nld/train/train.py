import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constant.config import DataConfig, TestConfig, TrainConfig
from ..constant.entities import (WANDB_ENTITY, WANDB_TESTING_PROJECT_NAME,
                                 WANDB_TRAINING_PROJECT_NAME)
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerDataset
from ..utils import clean_memory, compute_eer, set_random_seed_to, get_device


def train(
    cfg: TrainConfig, mislabeled_json_file: Optional[Path],
    vox1_mel_spectrogram_dir: Path, vox2_mel_spectrogram_dir: Path,
    training_model_save_dir: Path, save_interval: int,
    cuda_device_index: int, debug: bool
):
    set_random_seed_to(cfg.random_seed)
    device = get_device(cuda_device_index)

    if not debug:
        wandb.init(
            project=WANDB_TRAINING_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=cfg.to_dict(),
            name=cfg.description
        )

    checkpoint_dir = training_model_save_dir / cfg.description
    if not debug:
        checkpoint_dir.mkdir(exist_ok=True)
        cfg.to_json(checkpoint_dir / 'config.json')
        print(f'Checkpoints will be saved to `{checkpoint_dir}`')

    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    data_processing_config = DataConfig.from_json(vox2_mel_spectrogram_dir / 'data-processing-config.json')

    embedder_net = cfg.forge_model(data_processing_config.nmels, utterance_classes_num).to(device)
    criterion = cfg.forge_criterion(utterance_classes_num).to(device)

    train_dataset = SpeakerDataset(cfg.M, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file)
    train_data_loader = DataLoader(
        train_dataset,
        # Note that the real batch size = N * M
        batch_size=cfg.N,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': embedder_net.parameters()},
            {'params': criterion.parameters()}
        ], lr=cfg.learning_rate)
    else:
        raise ValueError(f'Unsupported module optimizer {cfg.optimizer}.')

    scaler = GradScaler()

    if (restore_model := cfg.restore_model_from) and (restore_loss := cfg.restore_loss_from):
        restore_model = Path(restore_model)
        restore_loss = Path(restore_loss)
        missing_keys, unexpected_keys = embedder_net.load_state_dict(torch.load(restore_model))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()
        missing_keys, unexpected_keys = criterion.load_state_dict(torch.load(restore_loss))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()

    if not debug:
        wandb.watch(embedder_net, criterion)

    embedder_net.train()
    total_iterations = 0
    clean_memory()
    with tqdm(total=cfg.iterations) as progress_bar:
        while total_iterations < cfg.iterations:
            total_loss = 0.0
            local_iterations = 0

            for mels, is_noisy, y, _, _ in train_data_loader:
                mels: Tensor = mels.to(device, non_blocking=True)
                is_noisy: Tensor = is_noisy.to(device)
                y: Tensor = y.to(device)

                assert mels.dim() == 4
                mels = mels.reshape((cfg.N * cfg.M, mels.size(2), mels.size(3)))

                optimizer.zero_grad()

                with autocast():
                    embeddings = embedder_net(mels)

                    if cfg.loss == 'GE2E':
                        embeddings = embeddings.reshape((cfg.N, cfg.M, embeddings.size(1)))
                        loss, _ = criterion(embeddings, y)
                    else:
                        assert cfg.loss in ('AAM', 'AAMSC', 'CE')
                        if cfg.loss != 'CE' and total_iterations == cfg.iterations // 8:
                            criterion.easy_margin = False
                        loss = criterion(embeddings, y.flatten())
                        if isinstance(loss, tuple):
                            loss, _ = loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss

                progress_bar.update(1)
                total_iterations += 1
                local_iterations += 1
                if total_iterations >= cfg.iterations:
                    break
                if (not debug) and total_iterations % save_interval == 0:
                    iteration_string = str(total_iterations).zfill(len(str(cfg.iterations)))
                    torch.save(embedder_net.state_dict(), checkpoint_dir / f'model-{iteration_string}.pth')
                    torch.save(criterion.state_dict(), checkpoint_dir / f'loss-{iteration_string}.pth')

            if not debug:
                wandb.log({'Loss': total_loss / local_iterations})

    if not debug:
        torch.save(embedder_net.state_dict(), checkpoint_dir / 'model-final.pth')
        torch.save(criterion.state_dict(), checkpoint_dir / 'loss-final.pth')
        wandb.finish()


def test(test_config: TestConfig, vox1test_mel_spectrogram_dir: Path, debug: bool):
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    training_config = TrainConfig.from_json(test_config.model_dir / 'config.json')
    set_random_seed_to(
        test_config.random_seed if test_config.random_seed is not None
        else training_config.random_seed
    )

    data_processing_config = DataConfig.from_json(vox1test_mel_spectrogram_dir / 'data-processing-config.json')

    embedder_net = training_config.forge_model(data_processing_config.nmels, VOX2_CLASS_NUM).to(device).eval()
    test_dataset = SpeakerDataset(test_config.M, None, vox1test_mel_spectrogram_dir, None)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_config.N,
        num_workers=test_config.dataloader_num_workers,
        collate_fn=test_dataset.collate_fn,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    missing_keys, unexpected_keys = embedder_net.load_state_dict(
        torch.load(test_config.model_dir / f'model-{test_config.iteration}.pth')
    )
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        raise ValueError()

    if not debug:
        wandb.init(
            project=WANDB_TESTING_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=test_config.to_dict(),
            name=f'{training_config.description}-{test_config.iteration}',
            reinit=True,
        )

    similarity_matrices = []
    labels = []
    for _ in tqdm(range(test_config.epochs), desc=f'Testing {test_config.iteration = }...'):
        for mels, _, _, _, _ in test_data_loader:
            mels: Tensor = mels.to(device)

            enrollment_batch, verification_batch = mels.split(int(mels.size(1) / 2), dim=1)
            enrollment_batch = enrollment_batch.reshape(
                (test_config.N * test_config.M // 2, enrollment_batch.size(2), enrollment_batch.size(3))
            )
            verification_batch = verification_batch.reshape(
                (test_config.N * test_config.M // 2, verification_batch.size(2), verification_batch.size(3))
            )

            enrollment_embeddings = embedder_net.get_embedding(enrollment_batch)
            verification_embeddings = embedder_net.get_embedding(verification_batch)

            enrollment_embeddings = enrollment_embeddings.reshape(
                (test_config.N, test_config.M // 2, enrollment_embeddings.size(1))
            )
            verification_embeddings = verification_embeddings.reshape(
                (test_config.N, test_config.M // 2, verification_embeddings.size(1))
            )

            enrollment_centroids = torch.mean(enrollment_embeddings, dim=1)
            # TODO: why direct reshape does not work?
            # verification_embeddings = verification_embeddings.reshape((-1, verification_embeddings.size(-1)))
            verification_embeddings = torch.cat(
                [verification_embeddings[:, 0], verification_embeddings[:, 1], verification_embeddings[:, 2]]
            )

            verification_embeddings_norm: Tensor = \
                verification_embeddings / torch.norm(verification_embeddings, dim=1).unsqueeze(-1)
            enrollment_centroids_norm: Tensor = \
                enrollment_centroids / torch.norm(enrollment_centroids, dim=1).unsqueeze(-1)

            similarity_matrix = verification_embeddings_norm @ enrollment_centroids_norm.transpose(-1, -2)
            ground_truth = torch.zeros(similarity_matrix.size())
            for i in range(ground_truth.size(0)):
                ground_truth[i, i % test_config.N] = 1

            similarity_matrices.append(similarity_matrix.flatten().detach().cpu().numpy())
            labels.append(ground_truth.flatten().detach().cpu().numpy())

        eer, thresh = compute_eer(
            np.concatenate(similarity_matrices),
            np.concatenate(labels),
        )

        if not debug:
            wandb.log({'Equal Error Rate': eer, 'Threshold': thresh})

    if not debug:
        wandb.finish()
