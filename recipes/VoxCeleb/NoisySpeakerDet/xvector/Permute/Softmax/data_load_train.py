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

def relabel(selected_file, utter_index, mislabel_dict):
    key = selected_file + "_{}".format(utter_index)
    if(key in mislabel_dict):
        return mislabel_dict[key]
    return None

class SpeakerDatasetPreprocessed(Dataset):
    
    def __init__(self, label_dict, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = 1
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)

        random.seed(time.time())
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        self.shuffle=shuffle
        self.utter_start = utter_start
        self.label_dict = label_dict
        with open("voxceleb2_20%_mislabel.json", "r") as f:
            self.mislabel_dict = json.load(f)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = self.file_list
        

        if self.shuffle:
            random.seed(time.time())
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]               
            
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]  
        else:
            utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]
        
        packet = relabel(selected_file[:-4], utter_index[0], self.mislabel_dict)

        if(packet is None): #non corrupted case
            label = self.label_dict[selected_file[:-4]]
        else:            
            label = self.label_dict[packet[0]]

        label = self.label_dict[selected_file[:-4]]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        
        #reshape from torch.Size([1, 160, 40]) to torch.Size([160, 40])
        utterance = torch.reshape(utterance, (utterance.size(1), utterance.size(2)))

        return utterance, label
