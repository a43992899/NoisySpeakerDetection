#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec
import time, json
from tqdm import tqdm

random.seed(1)
np.random.seed(1)

def relabel(selected_file, utter_index, mislabel_dict):
    key = selected_file + "_{}".format(utter_index)
    if(key in mislabel_dict):
        return mislabel_dict[key]
    return None

class SpeakerDatasetPreprocessed(Dataset):
    
    def __init__(self):
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.get_spkr_id()
        self.length = int(len(self.spkr2id)*1)
        self.get_spkr2utter()
    
    def get_spkr_id(self):
        speaker_list = os.listdir(self.path.replace("_single", ""))
        speaker_list.sort()
        self.spkr2id = {}
        self.id2spkr = {}
        for i in range(len(speaker_list)):
            self.spkr2id[speaker_list[i][:-4]] = i
            self.id2spkr[i] = speaker_list[i][:-4]
    
    def get_spkr2utter(self):
        """
        This function is used to get the correct and mislabeled spkr2utter dict
        spkr2utter format: {speaker_id: [file_name, ...]}, where file_name is all from the same speaker
        spkr2utter_mislabel format: {speaker_id: [file_name, ...]}, where some of the file_name are not from the corresponding speaker
        """
        self.file_list = os.listdir(self.path)
        self.file_list.sort()
        # mislabel_mapper maps a file to a wrong speaker
        with open("/home/yrb/code/speechbrain/Permute/Softmax/voxceleb2_20%_mislabel.json", "r") as f:
            mislabel_mapper = json.load(f)
        self.spkr2utter = {}
        self.spkr2utter_mislabel = {}
        print("Loading spkr2utter and spkr2utter_mislabel...")
        for file in tqdm(self.file_list):
            speaker_id = file.split("_")[0]
            if speaker_id not in self.spkr2utter:
                self.spkr2utter[speaker_id] = []
            self.spkr2utter[speaker_id].append(file)

            speaker_id = mislabel_mapper.get(file[:-4], speaker_id)
            if speaker_id not in self.spkr2utter_mislabel:
                self.spkr2utter_mislabel[speaker_id] = []
            self.spkr2utter_mislabel[speaker_id].append(file)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        selected_spkr = self.id2spkr[idx]
        if hp.training:
            utters = self.spkr2utter_mislabel[selected_spkr]
        else:
            utters = self.spkr2utter[selected_spkr]
        
        # select self.utter_num samples from utters
        selected_utters = random.sample(utters, self.utter_num)
        utterance_final, speaker_label, is_noisy, utterance_id = [], [], [], []
        for utter in selected_utters:
            data = np.load(os.path.join(self.path, utter))
            utterance_final.extend(data)
            speaker_label.extend([idx])
            if utter.split("_")[0] == selected_spkr:
                is_noisy.extend([0])
            else:
                is_noisy.extend([1])
            utterance_id.extend([utter])
        utterance = np.array(utterance_final)
        utterance = utterance[:,:,:160]
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1))) # transpose [batch, frames, n_mels]
        return utterance, np.array(speaker_label), np.array(is_noisy), utterance_id


class SpeakerDatasetPreprocessed_old(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        self.shuffle=shuffle
        self.utter_start = utter_start
        with open("/home/yrb/code/speechbrain/Permute/Softmax/voxceleb2_20%_mislabel.json", "r") as f:
            self.mislabel_dict = json.load(f)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)

        selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
            
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
       
        utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            
        utterance = utters[utter_index]  

        utterance_final = []
        for index in utter_index:
            packet = relabel(selected_file[:-4], index, self.mislabel_dict)
            if(packet is None): #not corrupted
                utterance_final.extend(utters[np.array([index])])
            else: #corrupted
                new_selected_file = packet[0] + ".npy"
                new_utters = np.load(os.path.join(self.path, new_selected_file))
                new_utter_index = np.array([packet[1]])
                new_utterance = new_utters[new_utter_index]
                utterance_final.extend(new_utterance)

        utterance = np.array(utterance_final)

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        
        return utterance
