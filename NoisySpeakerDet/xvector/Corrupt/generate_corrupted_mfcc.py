import os
import random
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import numpy as np
import json, glob, librosa, warnings
import concurrent.futures

warnings.filterwarnings("ignore")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_corrupted_mfcc(original, noise):
    result = original.copy()
    type_ = random.choice([True,False])
    if(type_): #replace
        start = random.choice(range(0, len(result) - len(noise) + 1))
        end = start + len(noise)
        result[start:end] = noise
    else: #add
        start = random.choice(range(0, len(result) - len(noise) + 1))
        end = start + len(noise)
        result[start:end] += noise
    return result

def get_random_noise_wav(min_duration):
    with open("/media/mnpham/Hard Disk 3/VoxCelebCorrupted/Experiment6/experiment6_noise.json", "r") as file:
        noise_lst = json.load(file)
    person = random.choice(noise_lst)[:-4]
    wav_lst = glob.glob("/media/mnpham/Hard Disk 2/VoxCeleb2/vox2_aac/dev/aac/{}/*/*".format(person))
    result = np.array([])
    while(len(result) == 0):
        selected_wav = random.choice(wav_lst)
        utter, sr = librosa.core.load(selected_wav, 16000)        # load utterance audio
        intervals = librosa.effects.split(utter, top_db=30)
        for interval in intervals:
            if (interval[1]-interval[0]) > min_duration:
                result = utter[interval[0]:interval[1]][:min_duration]
    return result

def save_corrupted_spectrogram_tisv_helper(speaker_lst, speaker_path, data_train_path, percentage_error):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length

    for i, folder in enumerate(speaker_lst):
        print("%dth speaker processing..."%i)
        utterances_spec = []
        folder = folder[:-4]
        print("speaker", folder)
        for video in os.listdir(os.path.join(speaker_path,folder)):
            for utter_name in os.listdir(os.path.join(speaker_path,folder, video)):
                if utter_name[-4:] == '.wav':
                    utter_path = os.path.join(speaker_path,folder, video, utter_name)         # path of each utterance
                    utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                    min_duration_error = int(28800*percentage_error)
                    intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
                    for interval in intervals:
                        if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long
                            
                            #first 180 frames
                            utter1 = utter
                            utter1_part = utter1[interval[0]:interval[1]][:28800]
                            noise1 = get_random_noise_wav(min_duration_error)
                            utter1_part = create_corrupted_mfcc(utter1_part, noise1)
                            S1 = librosa.core.stft(y=utter1_part, n_fft=hp.data.nfft,
                                                win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                            S1 = np.abs(S1)**2
                            mel_basis1 = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                            S1 = np.log10(np.dot(mel_basis1, S1) + 1e-6)  
                            utterances_spec.append(S1[:, :hp.data.tisv_frame])
#                             print(S1[:, :hp.data.tisv_frame].shape)
                            
                            #last 180 frames
                            utter2 = utter
                            utter2_part = utter2[interval[0]:interval[1]][-28800:]
                            noise2 = get_random_noise_wav(min_duration_error)
                            utter2_part = create_corrupted_mfcc(utter2_part, noise2)
                            S2 = librosa.core.stft(y=utter2_part, n_fft=hp.data.nfft,
                                                win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                            S2 = np.abs(S2)**2
                            mel_basis2 = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                            S2 = np.log10(np.dot(mel_basis2, S2) + 1e-6)  
                            utterances_spec.append(S1[:, -hp.data.tisv_frame:])
#                             print(S1[:, -hp.data.tisv_frame:].shape)
                            
        utterances_spec = np.array(utterances_spec)
        if True:      # save spectrogram as numpy file
            np.save(os.path.join(data_train_path, "corrupted_{}.npy".format(folder)), utterances_spec)

num_workers = 2

with open("/media/mnpham/Hard Disk 3/VoxCelebCorrupted/Experiment6/experiment6_train.json", "r") as file:
        speaker_lst = json.load(file)
        
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    for speaker_lst_partial in chunks(speaker_lst, int(len(speaker_lst)/num_workers)):
        futures.append(executor.submit(save_corrupted_spectrogram_tisv_helper, speaker_lst_partial, 
                                       "/media/mnpham/Hard Disk 2/VoxCeleb2/vox2_aac/dev/aac",
                                      "/media/mnpham/Hard Disk 3/VoxCelebCorrupted/Experiment6/CorruptedPercentage=0.75",
                                      0.75))
    for future in concurrent.futures.as_completed(futures):
        print("Done")