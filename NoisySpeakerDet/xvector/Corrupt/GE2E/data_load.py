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

#This function helps identify which "group" is corrupted
def get_corrupted_index(length):
    def divide_chunks(l, n):        
        # looping till length l 
        for i in range(0, len(l), n):  
            yield l[i:i + n]

    random.seed(time.time())
    chance = 1 #chance of a group being corrupted
    corrupted_index = []
    t = np.array(list(divide_chunks(list(range(length)), hp.train.N)))
    t_corrupted = t[np.random.permutation(len(t))[:int(chance*len(t))]] #list of minibatch that is corrupted
    for i in t_corrupted:
        for j in i:
            if(j % 2 != 0): 
                # j%2 for 2 corrupted groups per minibatch
                # j%4 for 3 corrupte groups per minibatch
                # j%1 for 4 corrupted groups per minibatch
                corrupted_index.append(j)
    return corrupted_index

class SpeakerDatasetPreprocessed(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            print(hp.data.train_path)
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M

        with open("/media/mnpham/Hard Disk 3/VoxCelebCorrupted/Experiment6/experiment6_train.json") as file_:
            data = json.load(file_)
        self.file_list = data #load the 3000 selected speaker for training
        
        random.seed(time.time())
        random.shuffle(self.file_list)
        self.length = int(len(self.file_list)*1)
        self.file_list = self.file_list[:self.length]
        self.shuffle=shuffle
        self.utter_start = utter_start
        self.corrupted_index = get_corrupted_index(self.length)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        np_file_list = self.file_list

        #total of 5 utterances
        num_utterance_corrupted = 3 #number of corrupted utterances
        num_utterance_normal = hp.train.M - num_utterance_corrupted #number of normal utterances

        #reset corrupted_index after an epoch
        if(idx == self.length - 3):
            self.corrupted_index = get_corrupted_index(self.length)

        if(idx in self.corrupted_index): #This "group" is corrupted
            utters_final = []
            random.seed(time.time())
            selected_file = random.sample(np_file_list, 1)[0]
            corrupted_selected_file = "corrupted_" + str(selected_file)
            
            #add normal utterance
            utters1 = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
            if self.shuffle:
                utter_index1 = np.random.randint(0, utters1.shape[0], num_utterance_normal)   # select utterances per speaker
                u1 = utters1[utter_index1]
            else:
                u1 = utters1[self.utter_start: self.utter_start+num_utterance_normal] # utterances of a speaker [batch(M), n_mels, frames]
            utters_final.extend(u1)

            #add corrupted utterance
            utters2 = np.load(os.path.join("/media/mnpham/Hard Disk 3/VoxCelebCorrupted/Experiment6/CorruptedPercentage=0.25", corrupted_selected_file)) #load a pre-generated corrupted utterance
            if self.shuffle:
                utter_index2 = np.random.randint(0, utters2.shape[0], num_utterance_corrupted)
                u2 = utters2[utter_index2]
            else:
                u2 = utters2[self.utter_start: self.utter_start+num_utterance_corrupted]
            utters_final.extend(u2)

            utters_final = np.array(utters_final)
            utterance = utters_final

        else: #This is a normal group
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

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        
        return utterance