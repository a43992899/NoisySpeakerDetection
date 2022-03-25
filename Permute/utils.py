#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018
@author: harry
"""
import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from hparam import hparam as hp

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_centroids(embeddings):
    """
    embeddings: [N, M, emb_size]
    N speakers, M utterances per speaker

    Returns: [N, emb_size]
    """
    return torch.mean(embeddings, dim=1)

def get_centroid(embeddings, speaker_id, utterance_id):
    """get centroid without corresponding utterance
    embeddings: [N, M, emb_size], N speakers, M utterances per speaker
    speaker_id: speaker index
    utterance_id: utterance index

    Returns: [emb_size]
    """
    speaker_embs = embeddings[speaker_id].detach().clone()
    speaker_embs[utterance_id] = 0
    return torch.sum(speaker_embs, dim=0) / (speaker_embs.size(0)-1)

def get_cossim(embeddings, centroids, cos):
    """compute cosine similarity matrix among all (utter_emb, centroid) pairs
    when calculating cossim between an utter_emb with self-centroid, we need to recompute self-centroid by excluding the utterance
    embeddings: [N, M, emb_size]
    centroids: [N, emb_size]
    cos: cosine similarity

    Returns: [N, M, N] N speakers, M utterances per speaker, N speaker centroids
    """
    N = embeddings.size(0)
    M = embeddings.size(1)
    emb_size = embeddings.size(2)
    cossim = torch.zeros(N, M, N)
    for i in range(N):
        cossim[:, :, i] = cos(embeddings, centroids[i])
    for speaker_id in range(N):
        for utter_id in range(M):
            self_exclude_centroid = get_centroid(embeddings, speaker_id, utter_id)
            cossim[speaker_id, utter_id, speaker_id] = cos(embeddings[speaker_id, utter_id], self_exclude_centroid)
    return cossim

def calc_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss

def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db

def compute_eer(ypreds, ylabels):
    ypreds = np.concatenate(ypreds)
    ylabels = np.concatenate(ylabels)

    fpr, tpr, thresholds = roc_curve(ylabels, ypreds, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))
    embeddings = torch.tensor([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]).to(torch.float).reshape(3,2,3)
    centroids = get_centroids(embeddings)
    cossim = get_cossim(embeddings, centroids)
    sim_matrix = w*cossim + b
    loss, per_embedding_loss = calc_loss(sim_matrix)