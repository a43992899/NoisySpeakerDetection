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

#This function helps identify which "group" is corrupted
def get_corrupted_index(length):
    #@param: length: length of dataset / number of speakers
    def divide_chunks(l, n):        
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n]

    random.seed(time.time())
    chance = 1 #chance that a minibatch will be corrupted
    corrupted_index = []
    t = np.array(list(divide_chunks(list(range(length)), hp.train.N)))
    t_corrupted = t[np.random.permutation(len(t))[:int(chance*len(t))]] #list of minibatch that is corrupted
    for i in t_corrupted:
        for j in i:
            if(j % 2 != 0): 
                # (j % 2 != 0) for 2 corrupted groups per minibatch
                # (j % 4 != 0) for 3 corrupted groups per minibatch
                # (j % 1 == 0) for 4 corrupted groups per minibatch
                corrupted_index.append(j)
    return corrupted_index

class SpeakerDatasetPreprocessed(Dataset):
    
    def __init__(self):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        
        random.seed(time.time())
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        self.corrupted_index = get_corrupted_index(self.length)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)
        group_num = hp.train.M

        if(hp.training):
            #reset corrupted_index after an epoch
            if(idx == self.length - 3):
                self.corrupted_index = get_corrupted_index(self.length)

            if(idx in self.corrupted_index): #This group is corrupted
                utters_final = []
                random.seed(time.time())
                selected_file = random.sample(np_file_list, group_num) #select 5 different speakers to get utterances from

                for i in selected_file: #select 5 different utterances for this "group"
                    filename = i
                    utters = np.load(os.path.join(self.path, filename))        # load utterance spectrogram of selected speaker
                    utter_index = np.random.randint(0, utters.shape[0], 1)   # select 1 utterance
                    u = utters[utter_index]
                
                    utters_final.extend(u)

                utters_final = np.array(utters_final)
                utterance = utters_final

            else: #This group is normal
        
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
