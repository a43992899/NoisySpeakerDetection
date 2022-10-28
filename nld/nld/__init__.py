from .ge2e_centroid import compute_and_save_ge2e_embedding_centroid
from .inconsistency import (compute_confidence_inconsistency,
                            compute_distance_inconsistency)


def nld_save_ge2e_embedding_centroid_main(args):
    compute_and_save_ge2e_embedding_centroid(
        args.model_dir, args.selected_iteration, args.vox1_mel_spectrogram_dir,
        args.vox2_mel_spectrogram_dir, args.mislabeled_json_dir, args.debug,
    )


def nld_distance_main(args):
    compute_distance_inconsistency(
        args.model_dir, args.selected_iteration, args.vox1_mel_spectrogram_dir,
        args.vox2_mel_spectrogram_dir, args.mislabeled_json_dir, args.debug
    )


def nld_confidence_main(args):
    compute_confidence_inconsistency(
        args.model_dir, args.selected_iteration, args.vox1_mel_spectrogram_dir,
        args.vox2_mel_spectrogram_dir, args.mislabeled_json_dir, args.debug
    )
