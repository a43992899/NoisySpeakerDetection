#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # change to current file path
import argparse
import random
import time, shutil
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from hparam import Hparam
from data_load import SpeakerDatasetPreprocessed
from speech_embedder_net import SpeechEmbedder, SpeechEmbedder_Softmax, GE2ELoss, GE2ELoss_, AngularPenaltySMLoss, AAMSoftmax,SubcenterArcMarginProduct, get_centroids, get_cossim
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np
from numpy.linalg import solve
import scipy.linalg
import scipy.stats
from tqdm import tqdm
from utils import compute_eer

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params

def get_criterion(device):
    if hp.train.loss == 'CE':
        criterion = torch.nn.NLLLoss()
    elif hp.train.loss == 'GE2E':
        criterion = GE2ELoss_(init_w=10.0, init_b=-5.0, loss_method='softmax').to(device)
    elif hp.train.loss == 'AAM':
        criterion = AAMSoftmax(hp.model.proj, 5994, scale=hp.train.s, margin=hp.train.m, easy_margin=True).to(device)
    elif hp.train.loss == 'AAMSC':
        criterion = SubcenterArcMarginProduct(hp.model.proj, 5994, s=hp.train.s, m=hp.train.m, K=hp.train.K).to(device)
    else:
        raise ValueError('Unknown loss')
    return criterion

def get_model(device):
    if hp.train.loss == 'CE':
        embedder_net = SpeechEmbedder_Softmax(hp=hp, num_classes=5994).to(device)
    else:
        embedder_net = SpeechEmbedder(hp).to(device)
    return embedder_net

def get_optimizer(model, criterion):
    if hp.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': criterion.parameters()}
                ], lr=hp.train.lr)
    elif hp.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
                    {'params': model.parameters()},
                    {'params': criterion.parameters()}
                ], lr=hp.train.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError('Unknown optimizer')
    return optimizer

def get_checkpoint_dir():
    loss_type = hp.train.loss
    assert hp.train.noise_type in ['Permute', 'Open', 'Mix'], 'Unknown noise type'
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
    return os.path.join(hp.train.checkpoint_dir, f"{hp.train.noise_type}", loss_type, f"{hp.train.noise_level}%", sub_folder)

def train(model_path):
    hp.train.checkpoint_dir = get_checkpoint_dir()
    print('Will save checkpoints to:', hp.train.checkpoint_dir)
    # create log
    if not hp.train.debug:
        os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
        log_dir = os.path.join(hp.train.checkpoint_dir, 'log')
        writer = SummaryWriter(log_dir)
        try:
            shutil.copy(args.cfg, log_dir)
        except shutil.SameFileError:
            print('Config file already exists in log directory, skip copying')

    # init
    device = torch.device(hp.device)
    
    train_dataset = SpeakerDatasetPreprocessed(hp)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True, pin_memory=True) 
    
    embedder_net = get_model(device)
    criterion = get_criterion(device)
    optimizer = get_optimizer(embedder_net, criterion)
    scaler = GradScaler()

    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(model_path))
        try:
            criterion.load_state_dict(torch.load(model_path.replace('ckpt_epoch', 'ckpt_criterion_epoch')))
            criterion.m = hp.train.m
            print('Loaded criterion')
        except:
            pass
        restored_epoch = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
    else:
        restored_epoch = 0
    
    embedder_net.train()
    iteration = 0

    for e in range(restored_epoch, hp.train.epochs):
        total_loss = 0
        for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(train_loader): 
            utterance_ids = np.array(utterance_ids).T
            mel_db_batch = mel_db_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            labels = labels.reshape(hp.train.N*hp.train.M)

            # random permute the batch
            perm = torch.randperm(hp.train.N*hp.train.M)
            unperm = torch.argsort(perm)

            mel_db_batch = mel_db_batch[perm]

            optimizer.zero_grad()
            with autocast():
                embeddings = embedder_net(mel_db_batch)
                embeddings = embeddings[unperm]
                if hp.train.loss in {'GE2E'}:
                    embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
                    loss, prec1 = criterion(embeddings) #wants (Speaker, Utterances, embedding)
                    if batch_id % 100 == 0:
                        print('batch acc:', prec1.item())
                elif hp.train.loss in {'AAM', 'AAMSC', 'CE'}:
                    if hp.train.loss != 'CE':
                        if e >= 200:
                            criterion.easy_margin = False
                    loss = criterion(embeddings, labels)
                    if type(loss) == tuple:
                        loss, prec1 = loss
                        if batch_id % 100 == 0:
                            print('batch acc:', prec1.item())  
                else:
                    raise ValueError('Unknown loss')
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                writer.add_scalar('Loss/train', loss, iteration)
                print(mesg)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

            criterion.eval().cpu()
            ckpt_criterion_filename = "ckpt_criterion_epoch_" + str(e+1) + ".pth"
            ckpt_criterion_path = os.path.join(hp.train.checkpoint_dir, ckpt_criterion_filename)
            torch.save(criterion.state_dict(), ckpt_criterion_path)
            criterion.to(device).train()

            print('Saved checkpoint to', ckpt_model_path)

    #save model
    embedder_net.eval().cpu()
    ckpt_model_filename = "ckpt_final.pth"
    ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
    torch.save(embedder_net.state_dict(), ckpt_model_path)
    
    print("\nDone, trained model saved at", ckpt_model_path)

def test(model_path):
    print(model_path)
    device = torch.device(hp.device)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    test_dataset = SpeakerDatasetPreprocessed(hp)

    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    try:
        embedder_net = SpeechEmbedder(hp).to(device)
        embedder_net.load_state_dict(torch.load(model_path))
    except:
        embedder_net = SpeechEmbedder_Softmax(hp=hp, num_classes=5994).to(device)
        embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    print("Number of params: ", get_n_params(embedder_net))
    
    avg_EER = 0
    
    ypreds = []
    ylabels = []

    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(tqdm(test_loader)):
            assert hp.test.M % 2 == 0

            utterance_ids = np.array(utterance_ids).T
            mel_db_batch = mel_db_batch.to(device)
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = torch.randperm(verification_batch.size(0))
            unperm = torch.argsort(perm)
                
            verification_batch = verification_batch[perm]
            # get embedder_net attribute
            if embedder_net.__class__.__name__ == 'SpeechEmbedder_Softmax':
                enrollment_embeddings = embedder_net.get_embedding(enrollment_batch)
                verification_embeddings = embedder_net.get_embedding(verification_batch)
            else:
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)

            verification_embeddings = verification_embeddings[unperm]
            
            enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            veri_embed = torch.cat([verification_embeddings[:,0],verification_embeddings[:,1],verification_embeddings[:,2]])
            
            veri_embed_norm = veri_embed/torch.norm(veri_embed, dim = 1).unsqueeze(-1)
            enrl_embed = torch.cat([enrollment_centroids]*1)
            enrl_embed_norm = enrl_embed/torch.norm(enrl_embed, dim = 1).unsqueeze(-1)
            sim_mat = torch.matmul(veri_embed_norm, enrl_embed_norm.transpose(-1, -2)).data.cpu().numpy()
            truth = np.ones_like(sim_mat)*(-1)
            for i in range(truth.shape[0]):
                truth[i, i%10] = 1
            ypreds.append(sim_mat.flatten())
            ylabels.append(truth.flatten())

        eer, thresh = compute_eer(ypreds, ylabels)
        print("eer:", eer, "threshold:", thresh)
    print(model_path, 'eval done.')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/config.yaml', help='config.yaml path')
    args = parser.parse_args()

    hp = Hparam(file=args.cfg)

    if hp.stage == 'train':
        print(f"Train {hp.train.noise_type} Experiment")
        train(hp.train.model_path)
    else:
        print("Test Experiment")
        test(hp.test.model_path)