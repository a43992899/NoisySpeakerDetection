from pathlib import Path
from typing import List

from .ge2e_centroid import compute_and_save_ge2e_embedding_centroid
from .inconsistence import (compute_confidence_inconsistency,
                            compute_distance_inconsistency)


def nld_save_ge2e_embedding_centroid_main(args):
    compute_and_save_ge2e_embedding_centroid(
        args.model_dir, args.selected_iteration, args.vox1_mel_spectrogram_dir,
        args.vox2_mel_spectrogram_dir, args.mislabeled_json_dir
    )


def nld_distance_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: List[str] = args.selected_iterations
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
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    debug: bool = args.debug

    for iteration in selected_iterations:
        compute_confidence_inconsistency(
            model_dir, iteration, vox1_mel_spectrogram_dir,
            vox2_mel_spectrogram_dir, mislabeled_json_dir, debug
        )
