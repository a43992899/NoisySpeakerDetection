import json
import os
import random
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..constant.config import DataConfig, NOISE_LEVELS, NOISE_TYPES, NoiseLevel
from ..utils import set_random_seed_to


def produce_mel_spectrogram(args):
    vox1_audio_dir: Path = args.vox1_audio_dir
    vox1test_audio_dir: Path = args.vox1test_audio_dir
    vox2_audio_dir: Path = args.vox2_audio_dir
    vox1_output_dir: Path = args.vox1_output_dir
    vox1test_output_dir: Path = args.vox1test_output_dir
    vox2_output_dir: Path = args.vox2_output_dir
    cfg = DataConfig(args.sr, args.nfft, args.window, args.hop, args.nmels, args.tisv_frame, args.silence_threshold)

    vox1_output_dir.mkdir(parents=True, exist_ok=True)
    vox1test_output_dir.mkdir(parents=True, exist_ok=True)
    vox2_output_dir.mkdir(parents=True, exist_ok=True)

    utterance_min_length = (cfg.tisv_frame * cfg.hop + cfg.window) * cfg.sr
    mel_basis = librosa.filters.mel(sr=cfg.sr, n_fft=cfg.nfft, n_mels=cfg.nmels)
    for dataset_dir, output_dir in zip(
        (vox1_audio_dir, vox1test_audio_dir, vox2_audio_dir),
        (vox1_output_dir, vox1test_output_dir, vox2_output_dir)
    ):
        speaker_to_id: Dict[str, int] = dict()
        for speaker_id, speaker in tqdm(
            enumerate(os.listdir(dataset_dir)),
            desc=f'Processing data from {dataset_dir} to {output_dir}...'
        ):
            speaker_dir = dataset_dir / speaker
            if not speaker_dir.is_dir():
                continue
            speaker_to_id[speaker] = speaker_id
            utterance_index = 0
            for video in os.listdir(speaker_dir):
                video_dir = speaker_dir / video
                if not video_dir.is_dir():
                    continue
                for audio in os.listdir(video_dir):
                    audio_file = video_dir / audio
                    if not audio_file.is_file():
                        continue
                    utterance, _ = librosa.core.load(audio_file, sr=cfg.sr)
                    intervals: npt.NDArray[np.int32] = librosa.effects.split(utterance, top_db=cfg.silence_threshold)
                    assert intervals.ndim == 2
                    assert intervals.shape[1] == 2
                    for i in range(intervals.shape[0]):
                        if (end_index := intervals[i, 1]) - (start_index := intervals[i, 0]) <= utterance_min_length:
                            continue
                        utterance_part = utterance[start_index:end_index]
                        spectrogram = librosa.core.stft(
                            utterance_part,
                            n_fft=cfg.nfft,
                            win_length=int(cfg.window * cfg.sr),
                            hop_length=int(cfg.hop * cfg.sr)
                        )
                        spectrogram = np.abs(spectrogram) ** 2
                        spectrogram = np.log10(np.dot(mel_basis, spectrogram) + 1e-6)

                        spec1 = spectrogram[np.newaxis, :, :cfg.tisv_frame]
                        np.save(output_dir / f'{speaker}-{utterance_index}.npy', spec1)
                        utterance_index += 1

                        spec2 = spectrogram[np.newaxis, :, -cfg.tisv_frame:]
                        np.save(output_dir / f'{speaker}-{utterance_index}.npy', spec2)
                        utterance_index += 1

        with open(output_dir / 'spkr2id.json', 'w') as f:
            json.dump(speaker_to_id, f)
        cfg.to_json(output_dir / 'data-processing-config.json')


def produce_noisy_label(args):
    set_random_seed_to(args.random_seed)
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_output_dir: Path = args.vox1_output_dir
    vox2_output_dir: Path = args.vox2_output_dir

    with open(vox1_output_dir / 'spkr2id.json', 'r') as vox2_mel_spectrogram:
        vox1_speaker_to_id: Dict[str, int] = json.load(vox2_mel_spectrogram)
        vox1_speaker_ids = list(vox1_speaker_to_id.keys())
    with open(vox2_output_dir / 'spkr2id.json', 'r') as vox2_mel_spectrogram_file:
        vox2_speaker_to_id: Dict[str, int] = json.load(vox2_mel_spectrogram_file)
        vox2_speaker_ids = list(vox2_speaker_to_id.keys())

    vox2_mel_spectrogram_files = sorted(vox2_output_dir.iterdir())
    for noise_level in NOISE_LEVELS:
        if noise_level == 0:
            continue
        for noise_type in NOISE_TYPES:
            print(f'Processing {noise_level = } and {noise_type = }')
            mislabeled_dict: Dict[str, str] = dict()

            for vox2_mel_spectrogram_file in vox2_mel_spectrogram_files:
                if random.random() <= noise_level / 100:
                    if noise_type == 'Permute':
                        mislabeled_dict[vox2_mel_spectrogram_file.stem] = random.choice(vox2_speaker_ids)
                    elif noise_type == 'Open':
                        mislabeled_dict[vox2_mel_spectrogram_file.stem] = random.choice(vox1_speaker_ids)
                    else:
                        assert noise_type == 'Mix'
                        mislabeled_dict[vox2_mel_spectrogram_file.stem] = \
                            (random.choice(vox1_speaker_ids) if random.random() <= 0.5
                                else random.choice(vox2_speaker_ids))

            with open(mislabeled_json_dir / noise_type / f'{noise_level}%.json', 'w') as f:
                json.dump(mislabeled_dict, f)
