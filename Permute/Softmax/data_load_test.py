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

# from hparam import hparam as hp
from utils import mfccs_and_spec
import time, json

class SpeakerDatasetPreprocessedTest(Dataset):
    
    def __init__(self, file_list, utter_num=5, shuffle=True, utter_start=0):
        
        # data path
        # if hp.training:
        #     print(hp.data.train_path)
        #     self.path = hp.data.train_path
        #     self.utter_num = hp.train.M
        # else:
        #     self.path = hp.data.test_path
        #     self.utter_num = hp.test.M


        self.utter_num = utter_num
        self.file_list = file_list
        
        random.seed(time.time())
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = self.file_list

        if self.shuffle:
            random.seed(time.time())
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]               
            
        utters = np.load(selected_file)        # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]  
        else:
            utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        
        return utterance
