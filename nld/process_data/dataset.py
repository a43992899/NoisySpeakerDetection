import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constant.config import Config


class SpeakerDataset(Dataset):
    utterance_dir: Path
    utterance_files: List[str]
    spkr2id: Dict[str, int]
    mislabeled_mapping: Optional[Dict[str, str]]

    def __init__(self, utterance_dir: Path, mislabeled_json_file: Optional[Path]) -> None:
        super().__init__()
        if not utterance_dir.exists():
            raise FileNotFoundError()
        if not utterance_dir.is_dir():
            raise NotADirectoryError()

        spkr2id_file = utterance_dir / 'spkr2id.json'
        if not spkr2id_file.exists():
            raise FileNotFoundError()
        if not spkr2id_file.is_file():
            raise IsADirectoryError()

        processing_config = utterance_dir / 'data-processing-config.json'
        if not processing_config.exists():
            raise FileNotFoundError()
        if not processing_config.is_file():
            raise IsADirectoryError()

        if mislabeled_json_file is not None:
            if not mislabeled_json_file.exists():
                raise FileNotFoundError()
            if not mislabeled_json_file.is_file():
                raise IsADirectoryError()

        self.utterance_dir = utterance_dir
        self.utterance_files = sorted(os.listdir(utterance_dir))
        with open(spkr2id_file, 'r') as f:
            self.spkr2id = json.load(f)
        if mislabeled_json_file is not None:
            with open(mislabeled_json_file, 'r') as f:
                self.mislabeled_mapping = json.load(f)

    def __len__(self):
        return len(self.utterance_files)

    def __getitem__(self, idx: int):
        selected_file = self.utterance_dir / self.utterance_files[idx]
        assert selected_file.suffix == '.npy'
        speaker_label = selected_file.stem.split('-')[0]
        speaker_id = self.spkr2id[speaker_label]
        if self.mislabeled_mapping is not None:
            speaker_id = self.mislabeled_mapping.get(speaker_label, speaker_id)
        # TODO: should we return multiple utterances?

    @staticmethod
    def collate_fn(batch):
        pass


class SpeakerDatasetPreprocessed(Dataset):
    def __init__(self, hp: Config):
        self.vox1_path = hp.data.vox1_path
        self.stage = hp.stage
        if self.stage == "train":
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
            self.noise_type = hp.train.noise_type
            self.noise_level = hp.train.noise_level
        elif self.stage == "test":  # test EER
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
            self.noise_type = None
            self.noise_level = 0
        elif self.stage == "nld":  # noise label detection
            self.path = hp.data.nld_path
            self.utter_num = hp.nld.M
            self.noise_type = hp.nld.noise_type
            self.noise_level = hp.nld.noise_level
        else:
            raise ValueError("stage should be train/test/nld")
        assert self.noise_type in ["Permute", "Open", "Mix", None], "noise_type should be Permute/Open/Mix/None"
        assert self.noise_level in [0, 20, 50, 75], "noise_level should be 0/20/50/75"
        self.get_spkr_id()
        self.length = int(len(self.spkr2id) * 1)
        self.get_spkr2utter()

    def isFilename(self, inp):
        # TODO: consider using os.path.isfile?
        os.path.isfile(inp)
        return '.' in inp

    def get_spkr_id(self):
        spkr2id_json = os.path.join(self.path, "../spkr2id.json")
        self.spkr2id = json.load(open(spkr2id_json, "r"))
        self.id2spkr = {v: k for k, v in self.spkr2id.items()}

    def get_spkr2utter(self):
        """
        This function is used to get the correct and mislabeled spkr2utter dict
        spkr2utter format: {speaker_id: [file_name, ...]}, where file_name is all from the same speaker
        spkr2utter_mislabel format: {speaker_id: [file_name, ...]}, where some of the file_name are not from the corresponding speaker
        """
        self.file_list = os.listdir(self.path)
        self.file_list.sort()
        if self.noise_level > 0:
            # mislabel_mapper maps a file to a wrong speaker
            with open(f"/home/yrb/code/speechbrain/data/jsons/{self.noise_type}/voxceleb2_{self.noise_level}%_mislabel.json", "r") as f:
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
            self.spkr2utter[speaker_id].append(os.path.join(self.path, file))

            # if noise level is 0, then spkr2utter = spkr2utter_mislabel
            speaker_id_or_filename = mislabel_mapper.get(file[:-4], speaker_id)
            if self.isFilename(speaker_id_or_filename):
                # a vox1 file name, for ood noise
                filename = speaker_id_or_filename
                filepath = os.path.join(self.vox1_path, filename)
                speaker_id = file.split("_")[0]
                if speaker_id not in self.spkr2utter_mislabel:
                    self.spkr2utter_mislabel[speaker_id] = []
                self.spkr2utter_mislabel[speaker_id].append(filepath)
            else:
                # a speaker id label, for permute noise / clean samples
                speaker_id = speaker_id_or_filename
                if speaker_id not in self.spkr2utter_mislabel:
                    self.spkr2utter_mislabel[speaker_id] = []
                self.spkr2utter_mislabel[speaker_id].append(os.path.join(self.path, file))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        selected_spkr = self.id2spkr[idx]
        if self.stage in ["train", "nld"]:
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
            data = np.load(utter)
            utterance_final.extend(data)
            speaker_label.extend([idx])
            utter_name = os.path.split(utter)[-1]
            if utter_name.split("_")[0] == selected_spkr:
                is_noisy.extend([0])
            else:
                is_noisy.extend([1])
            utterance_id.extend([utter_name])
        utterance = np.array(utterance_final)
        utterance = utterance[:, :, :160]
        utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
        return utterance, np.array(speaker_label), np.array(is_noisy), utterance_id
