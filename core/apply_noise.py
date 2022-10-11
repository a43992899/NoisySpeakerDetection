import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .config import Config as hp
from .utils import set_random_seed_to

# TODO: how did Ruibin generate all mel spectrogram?

VOX2_DB = Path("./data/voxceleb/vox2/spmel")
VOX1_DB = Path("./data/voxceleb/vox1/spmel")
VOX1_TEST_DB = Path("./data/voxceleb/vox1_test/spmel")

NOISE_LEVELS = [20, 50, 75]
NOISE_TYPES = ['permute', 'open', 'mix']


def apply_noise(args):
    vox1_db_path = Path(args.vox1_db_path)
    vox2_db_path = Path(args.vox2_db_path)

    for db_path in (vox1_db_path, vox2_db_path):
        db_path = Path(db_path)

        spmel_path = db_path / 'spmel'
        spmel_single_path = db_path / 'spmel_single'
        db_json_path = db_path / 'spkr2id.json'

        db_spmel_file_paths = sorted(spmel_path.iterdir())
        spkr2id_dict: Dict[str, int] = dict()

        for i, db_spmel_file_path in enumerate(db_spmel_file_paths):
            if db_spmel_file_path.suffix == '.npy':
                x: npt.NDArray[np.float32] = np.load(db_spmel_file_path)
                for j in range(x.shape[0]):
                    np.save(spmel_single_path / f'{db_spmel_file_path.stem}_{j}.npy', x[j:j + 1, :, :])
                spkr2id_dict[db_spmel_file_path.stem] = i

        with open(db_json_path, 'w') as db_json_file:
            json.dump(spkr2id_dict, db_json_file)

    spmel_single_file_list = sorted(spmel_single_path.iterdir())
    speaker_id_list = sorted(spkr2id_dict.keys())

    # TODO vox celeb 1 and 2 data passing

    for noise_type in NOISE_TYPES:
        for noise_level in NOISE_LEVELS:
            set_random_seed_to(noise_level)

            mislabel_dict: Dict[str, str] = dict()
            for f in spmel_single_file_list:
                if random.random() <= noise_level / 100:
                    if noise_type == 'permute':
                        mislabel_dict[f.stem] = random.sample(speaker_id_list, 1)[0]
                    elif noise_type == 'open':
                        # TODO open noise
                        mislabel_dict[f.stem] = NotImplemented
                    else:
                        mislabel_dict[f.stem] = random.sample(speaker_id_list, 1)[0] \
                            if random.random() <= 0.5 else NotImplemented

            with open(db_path / f'{noise_type}_{noise_level}%.json', 'w') as f:
                json.dump(mislabel_dict, f)


# set current working directory
os.chdir('/home/yrb/code/speechbrain/core/')
sys.path.append('/home/yrb/code/speechbrain/core/')
warnings.filterwarnings("ignore")


def save_spectrogram_spkr_reco(speaker_path: str, data_train_path: str):
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    # total_speaker_num = len(os.listdir(speaker_path))
    # train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    # print("total speaker number : %d"%total_speaker_num)
    # print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for folder in tqdm(os.listdir(speaker_path)):
        utterances_spec = []
        for video in os.listdir(os.path.join(speaker_path, folder)):
            for utter_name in os.listdir(os.path.join(speaker_path, folder, video)):

                utter_path = os.path.join(speaker_path, folder, video, utter_name)         # path of each utterance

                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio

                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection
                for interval in intervals:
                    if (interval[1] - interval[0]) > utter_min_len:           # If partial utterance is sufficient long
                        # save first and last 180 frames of spectrogram.
                        utter_part = utter[interval[0]:interval[1]]
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft, win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        np.save(os.path.join(data_train_path, "{}.npy".format(folder)), utterances_spec)


# if __name__ == "__main__":
#     # save_spectrogram_spkr_reco("/home/yrb/code/speechbrain/data/voxceleb/vox2/aac","/home/yrb/code/speechbrain/data/voxceleb/vox2/spmel")
#     # save_spectrogram_spkr_reco("/home/yrb/code/speechbrain/data/voxceleb/vox1/wav","/home/yrb/code/speechbrain/data/voxceleb/vox1/spmel")
#     save_spectrogram_esc50("/home/yrb/code/speechbrain/data/ESC-50/audio","/home/yrb/code/speechbrain/data/ESC-50/spmel")
