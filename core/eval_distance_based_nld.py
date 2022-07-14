import argparse
import os
import random
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import fit_bmm

from utils import get_all_file_with_ext, isTarget, write_to_csv
from hparam import Hparam
from data_load import SpeakerDatasetPreprocessed
from speech_embedder_net import SpeechEmbedder, SpeechEmbedder_Softmax, \
    GE2ELoss_, AAMSoftmax, SubcenterArcMarginProduct, get_cossim


os.chdir(os.path.dirname(os.path.abspath(__file__)))  # change to current file path

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
# print("current dir: ", os.getcwd())
hp = Hparam(file='config/config.yaml')
hp.stage = 'nld'
skip_plot_list = ['/home/yrb/code/speechbrain/data/models/Permute/GE2E/75%/m16_bs256/ckpt_epoch_100.pth']


def get_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params


def extract_emb(embedder_net: Union[SpeechEmbedder, SpeechEmbedder_Softmax], batch: Tensor):
    if batch.ndim == 4:
        batch = batch.reshape(-1, batch.size(2), batch.size(3))
    return embedder_net.get_embedding(batch)


def get_criterion(device, model_path):
    loss_type = None
    for i in ['/Softmax/', '/GE2E/', '/AAM/', '/AAMSC/']:
        if i in model_path:
            if i == '/Softmax/':
                loss_type = 'Softmax'
            elif i == '/GE2E/':
                loss_type = 'GE2E'
            elif i == '/AAM/':
                loss_type = 'AAM'
            else:
                loss_type = 'AAMSC'
                # get number k from string
                # "/home/yrb/code/speechbrain/data/models/Open/AAMSC/20%/m0.1_s15_k3_bs128/ckpt_epoch_40.pth"
                k = int(model_path.split('/')[-2].split('_')[-2].replace('k', ''))
            break
    if loss_type == 'Softmax':
        criterion = torch.nn.NLLLoss()
    elif loss_type == 'GE2E':
        criterion = GE2ELoss_(init_w=10.0, init_b=-5.0, loss_method='softmax').to(device)
    elif loss_type == 'AAM':
        criterion = AAMSoftmax(hp.model.proj, 5994, scale=hp.train.s, margin=hp.train.m, easy_margin=True).to(device)
    elif loss_type == 'AAMSC':
        criterion = SubcenterArcMarginProduct(hp.model.proj, 5994, s=hp.train.s, m=hp.train.m, K=k).to(device)
    else:
        raise ValueError('Unknown loss')
    return criterion, loss_type

# model_path = hp.nld.model_path


device = torch.device(hp.device)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def eval_one(model_path, stop_batch_id=500, skip_plot=False):
    # rewrite noise level and noise type
    noise_type, noise_level = model_path.split('/')[-5], model_path.split('/')[-3]
    noise_level = int(noise_level.replace('%', ''))
    hp.nld.noise_type = noise_type
    hp.nld.noise_level = noise_level

    # load model
    try:
        embedder_net = SpeechEmbedder(hp).to(device)
        embedder_net.load_state_dict(torch.load(model_path))
    except BaseException:
        embedder_net = SpeechEmbedder_Softmax(hp=hp, num_classes=5994).to(device)
        embedder_net.load_state_dict(torch.load(model_path))

    criterion = get_criterion(hp.device, model_path)[0]
    criterion.load_state_dict(torch.load(model_path.replace('ckpt_epoch', 'ckpt_criterion_epoch')))
    embedder_net.eval()
    criterion.eval()

    # print("Number of params: ", get_n_params(embedder_net))

    ypreds = []
    ylabels = []

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    nld_dataset = SpeakerDatasetPreprocessed(hp)
    nld_loader = DataLoader(
        nld_dataset,
        batch_size=hp.nld.N,
        shuffle=False,
        num_workers=hp.nld.num_workers,
        drop_last=False)
    for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(tqdm(nld_loader)):
        utterance_ids = np.array(utterance_ids).T
        mel_db_batch = mel_db_batch.to(device)

        embeddings = extract_emb(embedder_net, mel_db_batch)
        embeddings = torch.reshape(embeddings, (hp.nld.N, hp.nld.M, embeddings.size(1)))  # (1, M, 256)
        centroid = embeddings.mean(dim=1, keepdim=True)
        embeddings = embeddings / embeddings.norm(dim=2, keepdim=True)
        centroid = centroid / centroid.norm(dim=2, keepdim=True)
        cos_sim = get_cossim(embeddings, centroid, cos)
        ypreds.extend((1 - cos_sim).reshape(-1).cpu().detach().numpy().tolist())
        ylabels.extend(is_noisy.reshape(-1).cpu().detach().numpy().tolist())
        if batch_id == stop_batch_id:
            break

    # select top noise level % from ypreds
    noise_level = hp.nld.noise_level
    # print("noise level: ", hp.nld.noise_level)
    ypreds = np.array(ypreds)
    ylabels = np.array(ylabels)

    def compute_precision(ypreds, ylabels, noise_level):
        selected = np.argsort(ypreds)[-int(len(ypreds) * noise_level / 100):]
        selected_ypreds = ypreds[selected]
        selected_ylabels = ylabels[selected]
        # compute precision
        return selected_ylabels.sum() / len(selected_ylabels)
    detection_precision = compute_precision(ypreds, ylabels, noise_level)
    print("top noise level precision: ", detection_precision)

    # create df from ypreds and ylabels
    df = pd.DataFrame({"Distance": ypreds, "isNoisy": ylabels})
    # print("plot distance distribution")

    if not skip_plot:
        sns.set()
        distance_plot = sns.displot(df, x="Distance", hue="isNoisy")
        distance_plot.fig.set_dpi(100)
    else:
        distance_plot = None

    bmm_model, bmm_model_max, bmm_model_min = fit_bmm(ypreds, max_iters=50)
    # bmm_model.plot()

    # Noise level estimation
    if hp.nld.noise_level >= 70:
        estimated_noise_level = bmm_model.weight[0]
    else:
        estimated_noise_level = bmm_model.weight[1]
    real_noise_level = ylabels.sum() / len(ylabels) * 100
    noise_level_label = hp.nld.noise_level
    print("Estimated noise level: ", estimated_noise_level)
    print("Noise level: ", real_noise_level)
    # print("top estimated noise level precision: ", compute_precision(ypreds, ylabels, estimated_noise_level))
    print("top noise level precision: ", compute_precision(ypreds, ylabels, hp.nld.noise_level))

    return detection_precision, estimated_noise_level, real_noise_level, noise_level_label, distance_plot


def evaluation(model_dir, csv_path, stop_batch_id=500):
    if not os.path.exists(csv_path):
        csv_header_line = 'ModelPath,Noise_level_lab,DetPrecision,Estimated_noise_lvl,Real_noise_level\n'
        write_to_csv(csv_path, csv_header_line)
        evaled_model_paths = []
    else:
        # read csv, get the model paths
        csv_file = open(csv_path, 'r')
        csv_lines = csv_file.readlines()
        csv_file.close()
        evaled_model_paths = [line.split(',')[0] for line in csv_lines[1:]]

    pth_list = sorted(get_all_file_with_ext(hp.nld.model_path, '.pth'))
    for file in pth_list:
        if 'ckpt_criterion_epoch' in file or file in evaled_model_paths or 'bs128' in file:
            continue
        else:
            file_to_eval = None
            if '/GE2E/' in file:
                if isTarget(
                    file,
                    target_strings=[
                        'ckpt_epoch_100.pth',
                        'ckpt_epoch_200.pth',
                        'ckpt_epoch_300.pth',
                        'ckpt_epoch_400.pth',
                        'ckpt_epoch_800.pth']):
                    file_to_eval = file
                else:
                    continue
            elif isTarget(file, target_strings=['/Softmax/', '/AAM/', '/AAMSC/']):
                if 'bs128' in file and 'ckpt_epoch_1600.pth' in file:
                    file_to_eval = file
                elif 'bs256' in file and 'ckpt_epoch_3200.pth' in file:
                    file_to_eval = file
                else:
                    continue
            else:
                continue
            print()
            print("Evaluating model: ", file_to_eval)
            detection_precision, estimated_noise_level, real_noise_level, noise_level_label, distance_plot = eval_one(
                file_to_eval, stop_batch_id, skip_plot=file_to_eval in skip_plot_list)
            csv_line = f"{file_to_eval},{noise_level_label},{detection_precision*100},{estimated_noise_level*100},{real_noise_level}\n"
            write_to_csv(csv_path, csv_line)
            # get dir of file_to_eval
            if distance_plot is not None:
                distance_plot.savefig(os.path.join(os.path.dirname(file_to_eval), "distance_distribution.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        type=str,
        default='../data/distance_based_nld_results.csv',
        help='csv path for writing nld results')
    parser.add_argument('--stop_batch_id', type=int, default=500, help='stop batch id for evaluation')
    args = parser.parse_args()
    evaluation(hp.nld.model_path, args.csv, args.stop_batch_id)
