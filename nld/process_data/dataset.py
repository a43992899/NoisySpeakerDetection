import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SpeakerDataset(Dataset):
    sample_num: int
    vox1_mel_spectrogram_dir: Path
    vox2_mel_spectrogram_dir: Path

    speaker_label_to_utterances: Dict[str, List[str]]
    speaker_label_to_id: Dict[str, int]
    speaker_labels = List[str]

    mislabeled_mapping: Optional[Dict[str, str]]

    def __init__(
        self, sample_num: int, vox1_mel_spectrogram_dir: Path,
        vox2_mel_spectrogram_dir: Path, mislabeled_json_file: Optional[Path]
    ) -> None:
        super().__init__()

        self.sample_num = sample_num
        self.vox1_mel_spectrogram_dir = vox1_mel_spectrogram_dir
        self.vox2_mel_spectrogram_dir = vox2_mel_spectrogram_dir

        self.speaker_label_to_utterances = dict()
        for utterance_file in vox2_mel_spectrogram_dir.iterdir():
            if not utterance_file.suffix == '.npy':
                continue
            speaker_label = utterance_file.stem.split('-')[0]
            try:
                self.speaker_label_to_utterances[speaker_label].append(utterance_file.name)
            except KeyError:
                self.speaker_label_to_utterances[speaker_label] = [utterance_file.name]

        with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
            self.speaker_label_to_id = json.load(f)

        assert len(self.speaker_label_to_utterances) == len(self.speaker_label_to_id)
        self.speaker_labels = sorted(self.speaker_label_to_id.keys())

        if mislabeled_json_file is not None:
            with open(mislabeled_json_file, 'r') as f:
                self.mislabeled_mapping = json.load(f)

    def __len__(self):
        return len(self.speaker_labels)

    def __getitem__(self, idx: int):
        selected_speaker_label = self.speaker_labels[idx]

        selected_speaker_mel_spectrogram_files = random.sample(
            self.speaker_label_to_utterances[selected_speaker_label], self.sample_num
        )
        selected_speaker_mel_spectrogram_files_untainted = [
            self.vox2_mel_spectrogram_dir / file
            for file in selected_speaker_mel_spectrogram_files
        ]
        selected_labels = [selected_speaker_label for _ in range(self.sample_num)]
        selected_file_is_noisy = [False for _ in range(self.sample_num)]

        for i in range(self.sample_num):
            vox2_selected_file = selected_speaker_mel_spectrogram_files[i]
            if self.mislabeled_mapping is not None and vox2_selected_file in self.mislabeled_mapping:
                selected_file_is_noisy[i] = True
                mislabeled = self.mislabeled_mapping[vox2_selected_file]
                if mislabeled.endswith('.npy'):
                    selected_speaker_mel_spectrogram_files[i] = self.vox1_mel_spectrogram_dir / mislabeled
                else:
                    selected_labels[i] = mislabeled
                    selected_speaker_mel_spectrogram_files[i] = self.vox2_mel_spectrogram_dir / vox2_selected_file
            else:
                selected_speaker_mel_spectrogram_files[i] = self.vox2_mel_spectrogram_dir / vox2_selected_file

        selected_id = [self.speaker_label_to_id[label] for label in selected_labels]

        selected_speaker_mel_spectrogram = np.stack([
            np.load(file).transpose() for file in selected_speaker_mel_spectrogram_files
        ])

        return (
            torch.from_numpy(selected_speaker_mel_spectrogram),
            torch.tensor(selected_file_is_noisy),
            torch.tensor(selected_id),
            selected_speaker_mel_spectrogram_files,
            selected_speaker_mel_spectrogram_files_untainted,
        )

    @staticmethod
    def collate_fn(batch: Sequence[Tuple[Tensor, Tensor, Tensor, List[str], List[str]]]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            [b[3] for b in batch],
            [b[4] for b in batch],
        )
