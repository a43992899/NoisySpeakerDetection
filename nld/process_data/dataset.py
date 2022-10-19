import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

VOX2_CLASS_NUM = 5994


class SpeakerUtteranceDataset(Dataset):
    noise_mel_spectrogram_dir: Optional[Path]
    main_mel_spectrogram_dir: Path

    speaker_utterances: List[str]
    speaker_label_to_id: Dict[str, int]
    speaker_labels: List[str]
    mislabeled_mapping: Optional[Dict[str, str]]

    def __init__(
        self, noise_mel_spectrogram_dir: Optional[Path], main_mel_spectrogram_dir: Path,
        mislabeled_json_file: Optional[Path],
    ):
        super().__init__()

        self.noise_mel_spectrogram_dir = noise_mel_spectrogram_dir
        self.main_mel_spectrogram_dir = main_mel_spectrogram_dir

        if mislabeled_json_file is not None:
            with open(mislabeled_json_file, 'r') as f:
                self.mislabeled_mapping = json.load(f)
        else:
            self.mislabeled_mapping = None

        with open(main_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
            self.speaker_label_to_id = json.load(f)
        self.speaker_labels = sorted(self.speaker_label_to_id.keys())

        self.speaker_utterances = [
            p.name for p in main_mel_spectrogram_dir.iterdir() if p.suffix == '.npy'
        ]

    def __len__(self):
        return len(self.speaker_utterances)

    def __getitem__(self, index: int):
        selected_speaker_utterance = self.speaker_utterances[index]
        original_utterance_path = self.main_mel_spectrogram_dir / selected_speaker_utterance
        original_label = selected_speaker_utterance.split('-')[0]

        if self.mislabeled_mapping is not None and selected_speaker_utterance in self.mislabeled_mapping:
            assert self.noise_mel_spectrogram_dir is not None
            is_noisy = True
            mislabeled = self.mislabeled_mapping[selected_speaker_utterance]
            if mislabeled.endswith('.npy'):
                tainted_utterane_path = self.noise_mel_spectrogram_dir / mislabeled
                spectrogram = np.load(tainted_utterane_path).transpose()
                label = original_label
            else:
                tainted_utterane_path = original_utterance_path
                spectrogram = np.load(original_utterance_path)
                label = mislabeled
        else:
            is_noisy = False
            tainted_utterane_path = original_utterance_path
            spectrogram = np.load(original_utterance_path).transpose()
            label = original_label

        selected_id = self.speaker_label_to_id[label]
        return torch.from_numpy(spectrogram), is_noisy, selected_id, tainted_utterane_path, original_utterance_path

    @staticmethod
    def collate_fn(batch: Sequence[Tensor, bool, int, Path, Path]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch]),
            [b[3] for b in batch],
            [b[4] for b in batch],
        )


class SpeakerDataset(Dataset):
    sample_num: int
    noise_mel_spectrogram_dir: Optional[Path]
    main_mel_spectrogram_dir: Path

    speaker_label_to_utterances: Dict[str, List[str]]
    speaker_label_to_id: Dict[str, int]
    speaker_labels: List[str]

    mislabeled_mapping: Optional[Dict[str, str]]

    def __init__(
        self, sample_num: int, noise_mel_spectrogram_dir: Optional[Path],
        main_mel_spectrogram_dir: Path, mislabeled_json_file: Optional[Path]
    ) -> None:
        super().__init__()

        self.sample_num = sample_num
        self.noise_mel_spectrogram_dir = noise_mel_spectrogram_dir
        self.main_mel_spectrogram_dir = main_mel_spectrogram_dir

        self.speaker_label_to_utterances = dict()
        for utterance_file in main_mel_spectrogram_dir.iterdir():
            if not utterance_file.suffix == '.npy':
                continue
            speaker_label = utterance_file.stem.split('-')[0]
            try:
                self.speaker_label_to_utterances[speaker_label].append(utterance_file.name)
            except KeyError:
                self.speaker_label_to_utterances[speaker_label] = [utterance_file.name]

        with open(main_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
            self.speaker_label_to_id = json.load(f)

        assert len(self.speaker_label_to_utterances) == len(self.speaker_label_to_id)
        self.speaker_labels = sorted(self.speaker_label_to_id.keys())

        if mislabeled_json_file is not None:
            with open(mislabeled_json_file, 'r') as f:
                self.mislabeled_mapping = json.load(f)
        else:
            self.mislabeled_mapping = None

    def __len__(self):
        return len(self.speaker_labels)

    def __getitem__(self, idx: int):
        selected_speaker_label = self.speaker_labels[idx]

        selected_speaker_mel_spectrogram_files = random.sample(
            self.speaker_label_to_utterances[selected_speaker_label], self.sample_num
        )
        selected_speaker_mel_spectrogram_files_untainted = [
            self.main_mel_spectrogram_dir / file
            for file in selected_speaker_mel_spectrogram_files
        ]
        selected_labels = [selected_speaker_label for _ in range(self.sample_num)]
        selected_file_is_noisy = [False for _ in range(self.sample_num)]

        for i in range(self.sample_num):
            vox2_selected_file = selected_speaker_mel_spectrogram_files[i]
            if self.mislabeled_mapping is not None and vox2_selected_file in self.mislabeled_mapping:
                assert self.noise_mel_spectrogram_dir is not None
                selected_file_is_noisy[i] = True
                mislabeled = self.mislabeled_mapping[vox2_selected_file]
                if mislabeled.endswith('.npy'):
                    assert self.noise_mel_spectrogram_dir is not None
                    selected_speaker_mel_spectrogram_files[i] = self.noise_mel_spectrogram_dir / mislabeled
                else:
                    selected_labels[i] = mislabeled
                    selected_speaker_mel_spectrogram_files[i] = self.main_mel_spectrogram_dir / vox2_selected_file
            else:
                selected_speaker_mel_spectrogram_files[i] = self.main_mel_spectrogram_dir / vox2_selected_file

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
    def collate_fn(batch: Sequence[Tuple[Tensor, Tensor, Tensor, List[Path], List[Path]]]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            [b[3] for b in batch],
            [b[4] for b in batch],
        )
