#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hparam import hparam as hp
from data_load import SpeakerDatasetPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np
from numpy.linalg import solve
import scipy.linalg
import scipy.stats

random.seed(1)
np.random.seed(1)

def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params

def get_mel_db_batch_corrupted_split(x_batch, duplicate_num, percentage_chance):
    def divide_chunks(l, n): 
        
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n] 

    chance_list = int(percentage_chance*100)*[1] + int((1-percentage_chance)*100)*[0]
    if(random.choice(chance_list)):
        split_tensor = []
        for speaker in x_batch:
            split_tensor_item = []
            ind = np.array(list(divide_chunks(np.random.permutation(len(speaker)), hp.train.M)))
            for i in ind:
                split_tensor_item.append(speaker[i])
            split_tensor.append(split_tensor_item)

        mel_db_batch = []
        for i in range(len(x_batch) - duplicate_num): #split
            mel_db_batch.append(random.sample(split_tensor[i],1)[0])
        mel_db_batch.extend(random.sample(split_tensor[len(x_batch) - duplicate_num], duplicate_num)) #normal
        mel_db_batch = torch.stack(mel_db_batch)

        return mel_db_batch
    else:
        mel_db_batch = []
        for speaker in x_batch:
            mel_db_batch.append(torch.stack(random.sample(list(speaker), hp.train.M)))
        mel_db_batch = torch.stack(mel_db_batch)

        return mel_db_batch

def train(model_path):
    device = torch.device(hp.device)
    
    train_dataset = SpeakerDatasetPreprocessed()

    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=False, num_workers=0, drop_last=True) 
    
    embedder_net = SpeechEmbedder().to(device)

    ge2e_loss = GE2ELoss(device)
    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    embedder_net.train()
    iteration = 0

    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, mel_db_batch in enumerate(train_loader): 
            mel_db_batch = get_mel_db_batch_corrupted_split(mel_db_batch,2,1)
            mel_db_batch = mel_db_batch.to(device)
            # print(mel_db_batch.shape)
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()

    #save model
    embedder_net.eval().cpu()
    torch.save(embedder_net.state_dict(), hp.model.model_path)
    
    print("\nDone, trained model saved at", hp.model.model_path)

def test(model_path):
    device = torch.device(hp.device)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    test_dataset = SpeakerDatasetPreprocessed()

    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(model_path))
    embedder_net.eval()

    print("Number of params: ", get_n_params(embedder_net))
    
    avg_EER = 0
    
    ypreds = []
    ylabels = []

    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            
            assert hp.test.M % 2 == 0
            mel_db_batch = mel_db_batch.to(device)
            enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3)))
            verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3)))
            
            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
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
    ypreds = np.concatenate(ypreds)
    ylabels = np.concatenate(ylabels)

    fpr, tpr, thresholds = roc_curve(ylabels, ypreds, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(eer, thresh)

if __name__=="__main__":
    if hp.training:
        print("Train Merge Experiment")
        train(hp.model.model_path)
    else:
        print("Test Merge Experiment")
        path = "/media/mnpham/HARD_DISK_3/VoxCelebCorrupted/Experiment2/Duplicate=2&Chance=0.5/ckpt_epoch_20_batch_id_1498.pth"
        test(path)