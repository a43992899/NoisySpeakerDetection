from pathlib import Path
from typing import List

from .distance_inconsistency import distance_inconsistency_evaluation
from .confidence_inconsistency import confidence_inconsistency_evaluation


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
        distance_inconsistency_evaluation(
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
        confidence_inconsistency_evaluation(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )


def nld_loss_main(args):
    pass
