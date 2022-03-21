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
        if hp.stage == "train":
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
            self.noise_level = hp.train.noise_level
        elif hp.stage == "test": # test EER
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
            self.noise_level = 0
        elif hp.stage == "nld": # noise label detection
            self.path = hp.data.nld_path
            self.utter_num = hp.nld.M
            self.noise_level = hp.nld.noise_level
        else:
            raise ValueError("stage should be train/test/nld")
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
        assert self.noise_level in [0, 20, 40, 50 ,60, 75]
        if self.noise_level > 0:
            # mislabel_mapper maps a file to a wrong speaker
            with open(f"/home/yrb/code/speechbrain/data/jsons/Permute/voxceleb2_{self.noise_level}%_mislabel.json", "r") as f:
                mislabel_mapper = json.load(f)
        else:
            mislabel_mapper = {}
        
        self.spkr2utter = {}
        self.spkr2utter_mislabel = {}
        print("Loading spkr2utter and spkr2utter_mislabel...")
        for file in tqdm(self.file_list):
            speaker_id = file.split("_")[0]
            if speaker_id not in self.spkr2utter:
                self.spkr2utter[speaker_id] = []
            self.spkr2utter[speaker_id].append(file)

            # if noise level is 0, then spkr2utter = spkr2utter_mislabel
            speaker_id = mislabel_mapper.get(file[:-4], speaker_id)
            if speaker_id not in self.spkr2utter_mislabel:
                self.spkr2utter_mislabel[speaker_id] = []
            self.spkr2utter_mislabel[speaker_id].append(file)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        selected_spkr = self.id2spkr[idx]
        if hp.stage == "train" or hp.stage == "nld":
            utters = self.spkr2utter_mislabel[selected_spkr]
        else:
            utters = self.spkr2utter[selected_spkr]
        
        # select self.utter_num samples from utters
        if self.utter_num > 0:
            selected_utters = random.sample(utters, self.utter_num)
        else:
            selected_utters = utters
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

