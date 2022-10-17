import json
from pathlib import Path
from typing import List, Optional

import torch
import wandb
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constant.config import DataConfig, TestConfig, TrainConfig
from ..constant.entities import (WANDB_ENTITY, WANDB_TESTING_PROJECT_NAME,
                                 WANDB_TRAINING_PROJECT_NAME)
from ..process_data.dataset import SpeakerDataset
from ..utils import clean_memory, set_random_seed_to, compute_eer


def train(
    cfg: TrainConfig, mislabeled_json_file: Path, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, training_model_save_dir: Path, save_interval: int, debug: bool
):
    set_random_seed_to(cfg.random_seed)
    device = torch.device('cuda' if cuda_is_available() else 'cpu')

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


def test(
    test_config: TestConfig, model_dir: Path,
    selected_iterations: Optional[List[str]], vox1test_mel_spectrogram_dir: Path, debug: bool,
):
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    training_config = TrainConfig.from_json(model_dir / 'config.json')
    set_random_seed_to(training_config.random_seed)

    models = list(map(lambda i: model_dir / f'model-{i}.pth', selected_iterations))

    data_processing_config = DataConfig.from_json(vox1test_mel_spectrogram_dir / 'data-processing-config.json')

    embedder_net = training_config.forge_model(data_processing_config.nmels).to(device).eval()
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

    for iteration, model in zip(selected_iterations, models):
        missing_keys, unexpected_keys = embedder_net.load_state_dict(torch.load(model))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()

        if not debug:
            wandb.init(
                project=WANDB_TESTING_PROJECT_NAME,
                entity=WANDB_ENTITY,
                config=test_config.to_dict(),
                name=f'{training_config.description}-{iteration}',
                reinit=True,
            )

        similarity_matrices = []
        labels = []
        for _ in tqdm(range(test_config.epochs), desc=f'Testing {iteration = }...'):
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
                verification_embeddings = torch.cat(
                    [verification_embeddings[:, 0], verification_embeddings[:, 1], verification_embeddings[:, 2]]
                )

                verification_embeddings_norm: Tensor = \
                    verification_embeddings / torch.norm(verification_embeddings, dim=1).unsqueeze(-1)
                # enrollment_embeddings = torch.cat([enrollment_centroids])
                enrollment_embeddings_norm: Tensor = \
                    enrollment_embeddings / torch.norm(enrollment_centroids, dim=1).unsqueeze(-1)

                similarity_matrix = verification_embeddings_norm @ enrollment_embeddings_norm.transpose(-1, -2)
                ground_truth = torch.ones(similarity_matrix.size()) * -1
                for i in range(ground_truth.size(0)):
                    ground_truth[i, i % test_config.N] = 1

                similarity_matrices.append(similarity_matrix)
                labels.append(ground_truth)

            eer, thresh = compute_eer(
                torch.cat(similarity_matrices).detach().cpu().numpy(),
                torch.cat(labels).detach().cpu().numpy()
            )

            if not debug:
                wandb.log({
                    'Equal Error Rate': eer,
                    'Threshold': thresh
                })

        if not debug:
            wandb.finish()


# def test_one(hp: Config):
#     try:
#         embedder_net = SpeechEmbedder(hp).to(device)
#         embedder_net.load_state_dict(torch.load(model_path))
#     except BaseException:
#         embedder_net = SpeechEmbedder_Softmax(
#             hp=hp, num_classes=5994).to(device)
#         embedder_net.load_state_dict(torch.load(model_path))
#     embedder_net.eval()

#     model_parameters = filter(lambda p: p.requires_grad, embedder_net.parameters())
#     num_params = sum([np.prod(p.size()) for p in model_parameters])
#     print("Number of params: ", num_params)

#     ypreds = []
#     ylabels = []

#     for _ in range(hp.test.epochs):
#         for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(tqdm(test_loader)):
#             assert hp.test.M % 2 == 0

#             utterance_ids = np.array(utterance_ids).T
#             mel_db_batch = mel_db_batch.to(device)
#             enrollment_batch, verification_batch = torch.split(
#                 mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)
#             enrollment_batch = torch.reshape(
#                 enrollment_batch,
#                 (hp.test.N * hp.test.M // 2,
#                  enrollment_batch.size(2),
#                  enrollment_batch.size(3)))
#             verification_batch = torch.reshape(
#                 verification_batch,
#                 (hp.test.N * hp.test.M // 2,
#                  verification_batch.size(2),
#                  verification_batch.size(3)))

#             perm = torch.randperm(verification_batch.size(0))
#             unperm = torch.argsort(perm)

#             verification_batch = verification_batch[perm]
#             # get embedder_net attribute
#             if embedder_net.__class__.__name__ == 'SpeechEmbedder_Softmax':
#                 enrollment_embeddings = embedder_net.get_embedding(
#                     enrollment_batch)
#                 verification_embeddings = embedder_net.get_embedding(
#                     verification_batch)
#             else:
#                 enrollment_embeddings = embedder_net(enrollment_batch)
#                 verification_embeddings = embedder_net(verification_batch)

#             verification_embeddings = verification_embeddings[unperm]

#             enrollment_embeddings = torch.reshape(
#                 enrollment_embeddings,
#                 (hp.test.N,
#                  hp.test.M // 2,
#                  enrollment_embeddings.size(1)))
#             verification_embeddings = torch.reshape(
#                 verification_embeddings,
#                 (hp.test.N,
#                  hp.test.M // 2,
#                  verification_embeddings.size(1)))

#             enrollment_centroids = get_centroids(enrollment_embeddings)
#             veri_embed = torch.cat(
#                 [verification_embeddings[:, 0], verification_embeddings[:, 1], verification_embeddings[:, 2]])

#             veri_embed_norm = veri_embed / torch.norm(veri_embed, dim=1).unsqueeze(-1)
#             enrl_embed = torch.cat([enrollment_centroids] * 1)
#             enrl_embed_norm = enrl_embed / torch.norm(enrl_embed, dim=1).unsqueeze(-1)
#             sim_mat = torch.matmul(veri_embed_norm, enrl_embed_norm.transpose(-1, -2)).data.cpu().numpy()
#             truth = torch.ones_like(sim_mat) * (-1)
#             for i in range(truth.shape[0]):
#                 truth[i, i % 10] = 1
#             ypreds.append(sim_mat.flatten())
#             ylabels.append(truth.flatten())

#         eer, thresh = compute_eer(ypreds, ylabels)
#         print("eer:", eer, "threshold:", thresh)
#     print(model_path, 'eval done.')
#     return eer, thresh
