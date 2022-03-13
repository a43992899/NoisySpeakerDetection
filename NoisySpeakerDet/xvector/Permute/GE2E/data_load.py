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

random.seed(1)
np.random.seed(1)

def relabel(selected_file, utter_index, mislabel_dict):
    key = selected_file + "_{}".format(utter_index)
    if(key in mislabel_dict):
        return mislabel_dict[key]
    return None

class SpeakerDatasetPreprocessed(Dataset):
    
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
        with open("voxceleb2_50%_mislabel.json", "r") as f:
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
