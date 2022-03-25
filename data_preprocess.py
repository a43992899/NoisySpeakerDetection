#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import os, sys, glob
# set current working directory
os.chdir('/home/yrb/code/speechbrain/Permute/GE2E')
sys.path.append('/home/yrb/code/speechbrain/Permute/GE2E')

import librosa
import numpy as np
from hparam import hparam as hp
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# downloaded dataset path                                

def save_spectrogram_tisv(speaker_path, data_train_path):
    
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    # total_speaker_num = len(os.listdir(speaker_path))
    # train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    # print("total speaker number : %d"%total_speaker_num)
    # print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(tqdm(os.listdir(speaker_path))):
        # print(folder)
        # print("%dth speaker processing..."%i)
        utterances_spec = []
        for video in os.listdir(os.path.join(speaker_path,folder)):
            for utter_name in os.listdir(os.path.join(speaker_path,folder, video)):          
                
                utter_path = os.path.join(speaker_path,folder, video, utter_name)         # path of each utterance
                    
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                    
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                                win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        np.save(os.path.join(data_train_path, "{}.npy".format(folder)), utterances_spec)

if __name__ == "__main__":
    # save_spectrogram_tisv("/home/yrb/code/speechbrain/data/voxceleb/vox2/aac","/home/yrb/code/speechbrain/data/voxceleb/vox2/spmel")
    save_spectrogram_tisv("/home/yrb/code/speechbrain/data/voxceleb/vox1/wav","/home/yrb/code/speechbrain/data/voxceleb/vox1/spmel")