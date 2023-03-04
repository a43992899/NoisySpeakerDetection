import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

VOX2_CLASS_NUM = 5994


class SpeakerDataset(Dataset):
    sample_num: int
    ood_mel_dir: Optional[Path]
    main_mel_dir: Path
    mislabel_mapper: Optional[Dict[str, str]]

    spkr_name2id: Dict[str, int]
    spkr_id2name: Dict[int, str]
    spkr_name2utter: Dict[str, List[Path]]
    spkr_name2utter_mislabel: Optional[Dict[str, List[Path]]]

    def __init__(
        self, sample_num: int, ood_mel_dir: Optional[Path],
        main_mel_dir: Path, mislabel_json: Optional[Path],
        del_utter_list: Optional[Path] = None
    ) -> None:
        super().__init__()

        self.sample_num = sample_num
        self.ood_mel_dir = ood_mel_dir
        self.main_mel_dir = main_mel_dir
        with open(self.main_mel_dir / 'speaker-label-to-id.json', 'r') as f:
            self.spkr_name2id = json.load(f)
            self.spkr_id2name = {v: k for k, v in self.spkr_name2id.items()}
        
        self.mislabel_mapper = None
        if mislabel_json is not None:
            with open(mislabel_json, 'r') as f:
                self.mislabel_mapper = json.load(f)            
        
        self.del_utter_mapper = dict()
        if del_utter_list is not None:
            with open(del_utter_list, 'r') as f:
                self.del_utter_mapper = set([line.strip() for line in f.readlines()])

        self.get_spkr2utter()

        self.empty_spkr = set()
        self.not_empty_spkr = set()
        if del_utter_list is not None:
            self.get_empty_spkr()
    
    def get_empty_spkr(self):
        for spkr_name, utter_list in self.spkr_name2utter_mislabel.items():
            if len(utter_list) == 0:
                self.empty_spkr.add(spkr_name)
            else:
                self.not_empty_spkr.add(spkr_name)

    def get_spkr2utter(self):
        """
        This function is used to get the correct and mislabeled spkr2utter dict
        spkr2utter format: {speaker_id: [file_name, ...]}, where file_name is all from the same speaker
        spkr2utter_mislabel format: {speaker_id: [file_name, ...]},
        where some of the file_name are not from the corresponding speaker
        """
        self.spkr_name2utter = dict()
        self.spkr_name2utter_mislabel = dict()

        main_mel_fl = sorted(os.listdir(self.main_mel_dir))
        for utter_file in tqdm(main_mel_fl, desc='Loading spkr_name2utter and spkr_name2utter_mislabel...'):
            utter_file = self.main_mel_dir / Path(utter_file)
            if not utter_file.suffix == '.npy':
                continue
            spkr_name = utter_file.stem.split('-')[0]

            # get spkr_name2utter
            if spkr_name not in self.spkr_name2utter:
                self.spkr_name2utter[spkr_name] = []
            self.spkr_name2utter[spkr_name].append(utter_file)

            # get spkr_name2utter_mislabel
            if self.mislabel_mapper is not None:
                if utter_file.name in self.mislabel_mapper:
                    mislabel_file_or_label = self.mislabel_mapper[utter_file.name]
                    if mislabel_file_or_label.endswith('.npy'):  # Is a file. Is the open noise
                        assert self.ood_mel_dir is not None
                        mislabel_file = self.ood_mel_dir / mislabel_file_or_label
                        if spkr_name not in self.spkr_name2utter_mislabel:
                            self.spkr_name2utter_mislabel[spkr_name] = []
                        # delete utterance (after nld process)
                        if mislabel_file.name in self.del_utter_mapper:
                            continue
                        self.spkr_name2utter_mislabel[spkr_name].append(mislabel_file)
                    else:  # Is a label. Is the permute noise
                        mislabel_spkr_name = mislabel_file_or_label
                        if mislabel_spkr_name not in self.spkr_name2utter_mislabel:
                            self.spkr_name2utter_mislabel[mislabel_spkr_name] = []
                        # delete utterance (after nld process)
                        if utter_file.name in self.del_utter_mapper:
                            continue
                        self.spkr_name2utter_mislabel[mislabel_spkr_name].append(utter_file)
                else:
                    # add clean utter
                    if spkr_name not in self.spkr_name2utter_mislabel:
                        self.spkr_name2utter_mislabel[spkr_name] = []
                    # delete utterance (after nld process)
                    if utter_file.name in self.del_utter_mapper:
                        continue
                    self.spkr_name2utter_mislabel[spkr_name].append(utter_file)

        if self.mislabel_mapper is None:
            self.spkr_name2utter_mislabel = None
        else:
            assert len(self.spkr_name2utter_mislabel) == len(self.spkr_name2utter)

        assert len(self.spkr_name2utter) == len(self.spkr_name2id)

    def __len__(self):
        return len(self.spkr_name2id)

    def __getitem__(self, idx: int):
        selected_spkr = self.spkr_id2name[idx]

        # handle empty speaker (reassign)
        if selected_spkr in self.empty_spkr:
            selected_spkr = random.sample(self.not_empty_spkr, 1)[0]

        if self.spkr_name2utter_mislabel is None:
            all_utters_of_selected_spkr = self.spkr_name2utter[selected_spkr]
        else:
            all_utters_of_selected_spkr = self.spkr_name2utter_mislabel[selected_spkr]

        if self.sample_num != -1:
            selected_utter_paths = random.sample(all_utters_of_selected_spkr, self.sample_num)
        else:
            selected_utter_paths = all_utters_of_selected_spkr

        selected_len = len(selected_utter_paths)
        selected_mels, selected_ids, is_noisy, selected_utter_names = [], [], [], []
        for i in range(selected_len):
            mel = np.load(selected_utter_paths[i]).T
            selected_mels.append(mel)
            selected_ids.append(idx)
            utter_spkr_name = selected_utter_paths[i].stem.split('-')[0]
            is_noisy.append(utter_spkr_name != selected_spkr)
            selected_utter_names.append(selected_utter_paths[i].name)

        selected_mels = np.stack(selected_mels)

        return (
            torch.from_numpy(selected_mels),
            torch.tensor(is_noisy),
            torch.tensor(selected_ids),
            selected_utter_names,
            None,
        )

    @staticmethod
    def collate_fn(batch: Sequence[Tuple[Tensor, Tensor, Tensor, List[str], None]]):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            [b[3] for b in batch],  # TODO: unpack lists into one single list
            [b[4] for b in batch],
        )
