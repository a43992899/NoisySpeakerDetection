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

class SpeakerDatasetPreprocessed(Dataset):
    
    def __init__(self):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M*8
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        
        random.seed(time.time())
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)

        if(hp.training):
        
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
                        
            utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
                
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]       

            utterance = utterance[:,:,:160]               # TODO implement variable length batch size

            utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        
        else:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
                        
            utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
                
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index] 

            utterance = utterance[:,:,:160]

            utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))

        return utterance
