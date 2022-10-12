import os
import random
import shutil
import time

import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .constant.config import Config, Hparam
from .constant.entities import WANDB_ENTITY, WANDB_PROJECT
from .model.loss import (AAMSoftmax, GE2ELoss_, SubcenterArcMarginProduct,
                         get_centroids)
from .model.model import SpeechEmbedder
from .process_data.dataset import SpeakerDatasetPreprocessed
from .utils import (compute_eer, get_all_file_with_ext, isTarget,
                    set_random_seed_to, write_to_csv)


def train(hp: Config, cfg: str, enable_wandb: bool):
    if enable_wandb:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    if (noise_type := hp.train.noise_type) not in ['Permute', 'Open', 'Mix']:
        raise NotImplementedError(f'Unsupported noise type {noise_type}')
    assert hp.train.noise_type in ['Permute', 'Open', 'Mix'], 'Unknown noise type'

    # Obtain checkpoint saving directory
    loss_type = hp.train.loss
    if loss_type == 'CE':
        loss_type = "Softmax"
        sub_folder = f"bs{hp.train.N}"
    elif loss_type == 'GE2E':
        bs = hp.train.M * hp.train.N
        sub_folder = f"m{hp.train.M}_bs{bs}"
    elif loss_type == 'AAM':
        sub_folder = f"m{hp.train.m}_s{hp.train.s}_bs{hp.train.N}"
    elif loss_type == 'AAMSC':
        sub_folder = f"m{hp.train.m}_s{hp.train.s}_k{hp.train.K}_bs{hp.train.N}"
    else:
        raise ValueError('Unknown loss')
    hp.train.checkpoint_dir = os.path.join(
        hp.train.checkpoint_dir,
        f"{hp.train.noise_type}",
        loss_type,
        f"{hp.train.noise_level}%",
        sub_folder)
    print('Will save checkpoints to:', hp.train.checkpoint_dir)

    # Writing training log
    if not hp.train.debug:
        os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(hp.train.checkpoint_dir, 'log')
        writer = SummaryWriter(log_dir)
        try:
            shutil.copy(cfg, log_dir)
        except shutil.SameFileError:
            print('Config file already exists in log directory, skip copying')

    # Get the training device
    device = torch.device(hp.device)

    # Get the training dataset and data loader
    train_dataset = SpeakerDatasetPreprocessed(hp)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=hp.train.N,
        shuffle=True,
        num_workers=hp.train.num_workers,
        drop_last=True,
        pin_memory=True)

    # Get training model
    embedder_net = SpeechEmbedder(hp, should_softmax=hp.train.loss == 'CE', num_classes=5994).to(device)

    if hp.train.loss == 'CE':
        criterion = torch.nn.NLLLoss()
    elif hp.train.loss == 'GE2E':
        criterion = GE2ELoss_(
            init_w=10.0,
            init_b=-5.0,
            loss_method='softmax').to(device)
    elif hp.train.loss == 'AAM':
        criterion = AAMSoftmax(
            hp.model.proj,
            5994,
            scale=hp.train.s,
            margin=hp.train.m,
            easy_margin=True).to(device)
    elif hp.train.loss == 'AAMSC':
        criterion = SubcenterArcMarginProduct(
            hp.model.proj,
            5994,
            s=hp.train.s,
            m=hp.train.m,
            K=hp.train.K).to(device)
    else:
        raise NotImplementedError('Unknown loss')

    if hp.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': embedder_net.parameters()},
            {'params': criterion.parameters()}
        ], lr=hp.train.lr)
    elif hp.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': embedder_net.parameters()},
            {'params': criterion.parameters()}
        ], lr=hp.train.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError(f'Unsupported module optimizer {hp.train.optimizer}.')
    scaler = GradScaler()

    # If we are about to resume the progress of a previous training
    model_path = hp.train.model_path
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
        try:
            criterion.load_state_dict(
                torch.load(model_path.replace('ckpt_epoch', 'ckpt_criterion_epoch')))
            # criterion.m = hp.train.m
            print('Loaded criterion')
        except BaseException:
            ...
        restored_epoch = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
    else:
        restored_epoch = 0

    embedder_net.train()

    # Record training iteration
    iteration = 0

    ############################
    #      Start training!     #
    ############################

    for epoch in range(restored_epoch, hp.train.epochs):
        total_loss = 0
        for batch_id, (mel_db_batch, labels, is_noisy,
                       utterance_ids) in enumerate(train_data_loader):
            utterance_ids = np.array(utterance_ids).T
            mel_db_batch = mel_db_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mel_db_batch = torch.reshape(
                mel_db_batch,
                (hp.train.N * hp.train.M,
                 mel_db_batch.size(2),
                 mel_db_batch.size(3)))
            labels = labels.reshape(hp.train.N * hp.train.M)

            # random permute the batch
            perm = torch.randperm(hp.train.N * hp.train.M)
            unperm = torch.argsort(perm)

            mel_db_batch = mel_db_batch[perm]

            optimizer.zero_grad()

            with autocast():
                embeddings = embedder_net(mel_db_batch)
                embeddings = embeddings[unperm]
                if hp.train.loss in {'GE2E'}:
                    embeddings = torch.reshape(
                        embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
                    # wants (Speaker, Utterances, embedding)
                    loss, prec1 = criterion(embeddings)
                    if batch_id % 100 == 0:
                        print('batch acc:', prec1.item())
                elif hp.train.loss in {'AAM', 'AAMSC', 'CE'}:
                    if hp.train.loss != 'CE':
                        if epoch >= 200:
                            criterion.easy_margin = False
                    loss = criterion(embeddings, labels)
                    if isinstance(loss, tuple):
                        loss, prec1 = loss
                        if batch_id % 100 == 0:
                            print('batch acc:', prec1.item())
                else:
                    raise NotImplementedError(
                        f'Unsupported loss function {hp.train.loss}')

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = '{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n'.format(
                    time.ctime(),
                    epoch + 1,
                    batch_id + 1,
                    len(train_dataset) // hp.train.N,
                    iteration,
                    loss,
                    total_loss / (batch_id + 1))

                writer.add_scalar('Loss/train', loss, iteration)
                print(mesg)

        # Save the training result.
        if hp.train.checkpoint_dir is not None and (
                epoch + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = f"ckpt_epoch_{epoch + 1}.pth"
            ckpt_model_path = os.path.join(
                hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

            criterion.eval().cpu()
            ckpt_criterion_filename = f"ckpt_criterion_epoch_{epoch + 1}.pth"
            ckpt_criterion_path = os.path.join(
                hp.train.checkpoint_dir, ckpt_criterion_filename)
            torch.save(criterion.state_dict(), ckpt_criterion_path)
            criterion.to(device).train()

            print('Saved checkpoint to', ckpt_model_path)

    # Save model
    embedder_net.eval().cpu()
    ckpt_model_filename = "ckpt_final.pth"
    ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
    torch.save(embedder_net.state_dict(), ckpt_model_path)

    print("\nDone, trained model saved at", ckpt_model_path)


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


def main(args):
    set_random_seed_to(1)

    hp = Hparam(args.cfg)
    if hp.stage == 'train':
        print(f"Train {hp.train.noise_type} Experiment")
        train(hp, args.cfg)
    else:
        print("Test Experiment")
        test(hp, args.csv)
