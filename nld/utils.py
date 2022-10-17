import gc
import os
import random
from datetime import datetime, timezone
from typing import Generator, List

import numpy as np
import numpy.typing as npt
import torch
import torch.autograd as grad
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch import Tensor
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as cuda_is_available


def clean_memory():
    gc.collect()
    if cuda_is_available():
        empty_cuda_cache()


def set_random_seed_to(seed: int = 1):
    """Set random seed to all random library used in the project.

    This function set the random seed for Python standard library, NumPy, and PyTorch

    Args:
        seed (int, optional): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    if ext.startswith('.'):
        ext = '.' + ext
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
    when calculating cossim between an utter_emb with self-centroid, 
    we need to recompute self-centroid by excluding the utterance
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


def compute_eer(predictions: npt.NDArray, labels: npt.NDArray):
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def current_utc_time() -> str:
    """Return current time in UTC timezone as a string.
    Useful for logging ML experiments.
    """
    dtn = datetime.now(timezone.utc)
    return '-'.join(list(map(str, [
        dtn.year, dtn.month, dtn.day, dtn.hour, dtn.minute, dtn.second
    ])))


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
