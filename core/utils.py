#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018
@author: harry
"""
import os
from typing import Generator, List

import librosa
import numpy as np
import numpy.typing as npt
import torch
import torch.autograd as grad
import scipy.stats as stats
import matplotlib.pyplot as plt
from torch import Tensor
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def get_all_file_with_ext(path: str, ext: str) -> Generator[str, None, None]:
    """Recurse all files within a directory.

    This method involkes `os.walk` method to trace down all files
    not only in *this* directory but also in all subdirectories.

    Args:
        path (str): The root directory to start searching
        ext (str): The target file extension

    Yields:
        str: The absolute path to the file with the extension.
    """
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)


def isTarget(file: str, target_strings: List[str]) -> bool:
    for target_string in target_strings:
        if target_string in file:
            return True
    return False


def write_to_csv(csv_path: str, line: str):
    with open(csv_path, 'a') as f:
        f.write(line)


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


def get_centroids(embeddings: Tensor):
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
    return torch.sum(speaker_embs, dim=0) / (speaker_embs.size(0) - 1)


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
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum() + 1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss


def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

# TODO: unused function? Also, having reference to not existed `hp` variable?


def mfccs_and_spec(wav_file, wav_process=False, calc_mfccs=False, calc_mag_db=False):
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window

    # Cut silence and fix length
    if wav_process:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)

    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    # db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T

    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T

    return mfccs, mel_db, mag_db


def compute_eer(ypreds, ylabels):
    ypreds = np.concatenate(ypreds)
    ylabels = np.concatenate(ylabels)

    fpr, tpr, thresholds = roc_curve(ylabels, ypreds, pos_label=1)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

# CODE FOR BETA MIXTURE MODEL


def weighted_mean(x: npt.ArrayLike, w: npt.ArrayLike):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x: npt.ArrayLike, w: npt.ArrayLike):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def fit_bmm(data, normalization=True, rm_outliers=False, max_iters=10):
    bmm_model_max, bmm_model_min = None, None
    if normalization:
        if rm_outliers:
            # outliers detection
            max_perc = np.percentile(data, 95)
            min_perc = np.percentile(data, 5)
            data = data[(data <= max_perc) & (data >= min_perc)]

            bmm_model_max = max_perc
            bmm_model_min = min_perc + 10e-6
        else:
            bmm_model_max = data.max()
            bmm_model_min = data.min() + 10e-6

        data = (data - bmm_model_min) / (bmm_model_max - bmm_model_min + 1e-6)

        data[data >= 1] = 1 - 10e-4
        data[data <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=max_iters)
    bmm_model.fit(data)

    bmm_model.create_lookup(1)

    return bmm_model, bmm_model_max, bmm_model_min


if __name__ == "__main__":
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))

    embeddings = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0]]
    ).to(torch.float).reshape(3, 2, 3)

    centroids = get_centroids(embeddings)
    cossim = get_cossim(embeddings, centroids)
    sim_matrix = w * cossim + b
    loss, per_embedding_loss = calc_loss(sim_matrix)
