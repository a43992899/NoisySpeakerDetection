import gc
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wandb
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constant.config import Config, NewTrainConfig
from ..constant.entities import WANDB_ENTITY, WANDB_PROJECT
from ..model.loss import (AAMSoftmax, GE2ELoss_, SubcenterArcMarginProduct,
                          get_centroids)
from ..model.model import SpeechEmbedder
from ..process_data.dataset import SpeakerDataset, SpeakerDatasetPreprocessed
from ..utils import (compute_eer, current_utc_time, get_all_file_with_ext,
                     isTarget, set_random_seed_to, write_to_csv)


def train(
    cfg: NewTrainConfig, mislabeled_json_file: Path, utterance_dir: Path,
    training_model_save_dir: Path, enable_wandb: bool
):
    set_random_seed_to(cfg.random_seed)
    job_name = current_utc_time()
    device = torch.device('cuda' if cuda_is_available() else 'cpu')

    if enable_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=cfg.to_dict(),
            name=job_name
        )

    model_dir = training_model_save_dir / job_name
    print(f'Checkpoint saving dir: {model_dir}')
    model_dir.mkdir()
    cfg.to_json(model_dir / 'config.json')

    spkr2id_file = utterance_dir / 'spkr2id.json'
    with open(spkr2id_file, 'r') as f:
        spkr2id: Dict[str, int] = json.load(f)
    utterance_classes_num = len(spkr2id)  # TODO: assert this to be 5994!!!
    embedder_net = SpeechEmbedder(should_softmax=(cfg.loss == 'CE'), num_classes=utterance_classes_num).to(device)

    # TODO: check loss related hyperparameters and create loss function
    if cfg.loss == 'CE':
        criterion = torch.nn.NLLLoss()
    elif cfg.loss == 'GE2E':
        criterion = GE2ELoss_(loss_method='softmax')
    elif cfg.loss == 'AAM':
        criterion = AAMSoftmax(
            cfg.model_projection_size, utterance_classes_num,
            scale=cfg.s, margin=cfg.m, easy_margin=True
        )
    elif cfg.loss == 'AAMSC':
        criterion = SubcenterArcMarginProduct(
            cfg.model_projection_size, utterance_classes_num,
            s=cfg.s, m=cfg.m, K=cfg.K
        )
    else:
        raise NotImplementedError('Unknown loss')
    criterion.to(device)

    train_dataset = SpeakerDataset(utterance_dir, mislabeled_json_file)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.N,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
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
        if not restore_model.exists():
            raise FileNotFoundError()
        if not restore_model.is_file():
            raise IsADirectoryError()
        if not restore_model.suffix == '.pth':
            raise ValueError()

        restore_loss = Path(restore_loss)
        if not restore_loss.exists():
            raise FileNotFoundError()
        if not restore_loss.is_file():
            raise IsADirectoryError()
        if not restore_loss.suffix == '.pth':
            raise ValueError()

        missing_keys, unexpected_keys = embedder_net.load_state_dict(torch.load(restore_model))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()
        missing_keys, unexpected_keys = criterion.load_state_dict(torch.load(restore_loss))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()

    gc.collect()
    if cuda_is_available():
        empty_cuda_cache()

    if enable_wandb:
        wandb.watch(embedder_net, criterion)

    embedder_net.train()
    # for epoch in range(restored_epoch, hp.train.epochs):
    for epoch in range(cfg.epoches):
        total_loss = 0
        for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(train_data_loader):
            utterance_ids = np.array(utterance_ids).T
            mel_db_batch = mel_db_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mel_db_batch = torch.reshape(
                mel_db_batch,
                (cfg.N * cfg.M, mel_db_batch.size(2), mel_db_batch.size(3))
            )
            labels = labels.reshape(cfg.N * cfg.M)

            perm = torch.randperm(cfg.N * cfg.M)
            unperm = torch.argsort(perm)

            mel_db_batch = mel_db_batch[perm]

            optimizer.zero_grad()

            with autocast():
                embeddings = embedder_net(mel_db_batch)
                embeddings = embeddings[unperm]
                if cfg.loss == 'GE2E':
                    embeddings = torch.reshape(embeddings, (cfg.N, cfg.M, embeddings.size(1)))
                    loss, prec1 = criterion(embeddings)
                else:
                    assert cfg.loss in ('AAM', 'AAMSC', 'CE')
                    if cfg.loss != 'CE' and epoch == 200:
                        criterion.easy_margin = False
                    loss = criterion(embeddings, labels)
                    if isinstance(loss, tuple):
                        loss, prec1 = loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss

            if enable_wandb:
                # TODO log data
                pass

        if (epoch + 1) % cfg.checkpoint == 0:
            embedder_net.eval().cpu()
            torch.save(embedder_net.state_dict(), model_dir / f'model-epoch-{epoch + 1}.pth')
            embedder_net.to(device).train()

            criterion.eval().cpu()
            torch.save(criterion.state_dict(), model_dir / f'loss-epoch-{epoch + 1}.pth')
            criterion.to(device).train()

    embedder_net.eval().cpu()
    torch.save(embedder_net.state_dict(), model_dir / 'ckpt-final.pth')

    if enable_wandb:
        wandb.finish()


def test_one(hp: Config):
    model_path = hp.test.model_path
    print(model_path)
    device = torch.device(hp.device)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    test_dataset = SpeakerDatasetPreprocessed(hp)

    test_loader = DataLoader(
        test_dataset,
        batch_size=hp.test.N,
        shuffle=True,
        num_workers=hp.test.num_workers,
        drop_last=True)

    try:
        embedder_net = SpeechEmbedder(hp).to(device)
        embedder_net.load_state_dict(torch.load(model_path))
    except BaseException:
        embedder_net = SpeechEmbedder_Softmax(
            hp=hp, num_classes=5994).to(device)
        embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    model_parameters = filter(lambda p: p.requires_grad, embedder_net.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of params: ", num_params)

    ypreds = []
    ylabels = []

    for _ in range(hp.test.epochs):
        for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(tqdm(test_loader)):
            assert hp.test.M % 2 == 0

            utterance_ids = np.array(utterance_ids).T
            mel_db_batch = mel_db_batch.to(device)
            enrollment_batch, verification_batch = torch.split(
                mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)
            enrollment_batch = torch.reshape(
                enrollment_batch,
                (hp.test.N * hp.test.M // 2,
                 enrollment_batch.size(2),
                 enrollment_batch.size(3)))
            verification_batch = torch.reshape(
                verification_batch,
                (hp.test.N * hp.test.M // 2,
                 verification_batch.size(2),
                 verification_batch.size(3)))

            perm = torch.randperm(verification_batch.size(0))
            unperm = torch.argsort(perm)

            verification_batch = verification_batch[perm]
            # get embedder_net attribute
            if embedder_net.__class__.__name__ == 'SpeechEmbedder_Softmax':
                enrollment_embeddings = embedder_net.get_embedding(
                    enrollment_batch)
                verification_embeddings = embedder_net.get_embedding(
                    verification_batch)
            else:
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)

            verification_embeddings = verification_embeddings[unperm]

            enrollment_embeddings = torch.reshape(
                enrollment_embeddings,
                (hp.test.N,
                 hp.test.M // 2,
                 enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(
                verification_embeddings,
                (hp.test.N,
                 hp.test.M // 2,
                 verification_embeddings.size(1)))

            enrollment_centroids = get_centroids(enrollment_embeddings)
            veri_embed = torch.cat(
                [verification_embeddings[:, 0], verification_embeddings[:, 1], verification_embeddings[:, 2]])

            veri_embed_norm = veri_embed / \
                torch.norm(veri_embed, dim=1).unsqueeze(-1)
            enrl_embed = torch.cat([enrollment_centroids] * 1)
            enrl_embed_norm = enrl_embed / \
                torch.norm(enrl_embed, dim=1).unsqueeze(-1)
            sim_mat = torch.matmul(
                veri_embed_norm, enrl_embed_norm.transpose(-1, -2)).data.cpu().numpy()
            truth = np.ones_like(sim_mat) * (-1)
            for i in range(truth.shape[0]):
                truth[i, i % 10] = 1
            ypreds.append(sim_mat.flatten())
            ylabels.append(truth.flatten())

        eer, thresh = compute_eer(ypreds, ylabels)
        print("eer:", eer, "threshold:", thresh)
    print(model_path, 'eval done.')
    return eer, thresh


def test(hp: Config, csv_path: str):
    if os.path.isfile(hp.test.model_path):
        test_one(hp)
    elif os.path.isdir(hp.test.model_path):
        if not os.path.exists(csv_path):
            csv_header_line = 'ModelPath,EER(%),Threshold(%)\n'
            write_to_csv(csv_path, csv_header_line)
            tested_model_paths = []
        else:
            # read csv, get the model paths
            csv_file = open(csv_path, 'r')
            csv_lines = csv_file.readlines()
            csv_file.close()
            tested_model_paths = [line.split(',')[0] for line in csv_lines[1:]]

        pth_list = sorted(get_all_file_with_ext(hp.test.model_path, '.pth'))
        for file in pth_list:
            if 'ckpt_criterion_epoch' in file or file in tested_model_paths:
                continue
            else:
                file_to_test = None
                if '/GE2E/' in file:
                    if isTarget(
                        file,
                        target_strings=[
                            'ckpt_epoch_100.pth',
                            'ckpt_epoch_200.pth',
                            'ckpt_epoch_300.pth',
                            'ckpt_epoch_400.pth',
                            'ckpt_epoch_800.pth']):
                        file_to_test = file
                    else:
                        continue
                elif isTarget(file, target_strings=['/Softmax/', '/AAM/', '/AAMSC/']):
                    if 'bs128' in file and 'ckpt_epoch_1600.pth' in file:
                        file_to_test = file
                    elif 'bs256' in file and 'ckpt_epoch_3200.pth' in file:
                        file_to_test = file
                    else:
                        continue
                else:
                    continue
                eer, thresh = test_one(file_to_test)
                csv_line = f"{file_to_test},{eer*100},{thresh*100}\n"
                write_to_csv(csv_path, csv_line)

    else:
        print("model_path is not a file or a directory")
