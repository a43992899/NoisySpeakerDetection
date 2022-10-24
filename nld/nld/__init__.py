from pathlib import Path
from typing import List

from .inconsistence import (compute_confidence_inconsistency,
                            compute_distance_inconsistency,
                            compute_loss_inconsistency)


def nld_distance_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: List[str] = args.selected_iterations
    if selected_iterations is None:
        selected_iterations = ['final']
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    for iteration in selected_iterations:
        compute_distance_inconsistency(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )


def nld_confidence_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: List[str] = args.selected_iterations
    if selected_iterations is None:
        selected_iterations = ['final']
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    for iteration in selected_iterations:
        compute_confidence_inconsistency(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )


def nld_loss_main(args):
    model_dir: Path = args.model_dir
    sampling_interval: int = args.sampling_interval
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    compute_loss_inconsistency(
        model_dir, sampling_interval, vox1_mel_spectrogram_dir,
        vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
    )
