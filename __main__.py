from argparse import ArgumentParser
from pathlib import Path

from nld.constant.config import LOSSES, NOISE_LEVELS, NOISE_TYPES, OPTIMIZERS
from nld.constant.defaults import *
from nld.nld import nld_distance_main, nld_loss_main, nld_confidence_main
from nld.process_data import produce_mel_spectrogram, produce_noisy_label
from nld.train import test_main, train_main

if __name__ != '__main__':
    raise RuntimeError()

main_parser = ArgumentParser()
main_parser.set_defaults(main_func=main_parser.print_help)
main_subparser = main_parser.add_subparsers()

produce_mel_spectrogram_parser = main_subparser.add_parser('produce-mel-spectrogram')
produce_mel_spectrogram_parser.set_defaults(main_func=produce_mel_spectrogram)
produce_mel_spectrogram_parser.add_argument('--vox1-audio-dir', type=Path, default=DEFAULT_VOX1_AUDIO_DIR)
produce_mel_spectrogram_parser.add_argument('--vox1test-audio-dir', type=Path, default=DEFAULT_VOX1TEST_AUDIO_DIR)
produce_mel_spectrogram_parser.add_argument('--vox2-audio-dir', type=Path, default=DEFAULT_VOX2_AUDIO_DIR)
produce_mel_spectrogram_parser.add_argument('--vox1-output-dir', type=Path, default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR)
produce_mel_spectrogram_parser.add_argument(
    '--vox1test-output-dir', type=Path, default=DEFAULT_VOX1TEST_MEL_SPECTROGRAM_DIR)
produce_mel_spectrogram_parser.add_argument('--vox2-output-dir', type=Path, default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR)
produce_mel_spectrogram_parser.add_argument('--sr', type=int, default=DEFAULT_SAMPLE_RATE)
produce_mel_spectrogram_parser.add_argument('--nfft', type=int, default=DEFAULT_NFFT)
produce_mel_spectrogram_parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_LENGTH)
produce_mel_spectrogram_parser.add_argument('--hop', type=float, default=DEFAULT_HOP_SIZE)
produce_mel_spectrogram_parser.add_argument('--nmels', type=int, default=DEFAULT_NMELS)
produce_mel_spectrogram_parser.add_argument('--tisv-frame', type=int, default=DEFAULT_TISV_FRAME)
produce_mel_spectrogram_parser.add_argument('--silence-threshold', type=int, default=DEFAULT_SILENCE_THRESHOLD)

produce_noisy_label_parser = main_subparser.add_parser('produce-noisy-label')
produce_noisy_label_parser.set_defaults(main_func=produce_noisy_label)
produce_noisy_label_parser.add_argument(
    '--mislabeled-json-dir', type=Path, default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR)
produce_noisy_label_parser.add_argument(
    '--vox1-mel-spectrogram-dir', type=Path, default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR)
produce_noisy_label_parser.add_argument(
    '--vox2-mel-spectrogram-dir', type=Path, default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR)
produce_noisy_label_parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)


train_parser = main_subparser.add_parser('train')
train_parser.set_defaults(main_func=train_main)
train_parser.add_argument('loss', choices=LOSSES, type=str)
train_parser.add_argument('noise_level', choices=NOISE_LEVELS, type=int)
train_parser.add_argument('noise_type', choices=NOISE_TYPES, type=str)
train_parser.add_argument('--N', default=-1, type=int)
train_parser.add_argument('--M', default=-1, type=int)
train_parser.add_argument('--s', default=-1, type=float)
train_parser.add_argument('--m', default=-1, type=float)
train_parser.add_argument('--K', default=-1, type=int)
train_parser.add_argument('--iterations', default=DEFAULT_ITERATIONS, type=int)
train_parser.add_argument('--optimizer', choices=OPTIMIZERS, default='Adam', type=str)
train_parser.add_argument('--learning-rate', default=DEFAULT_LEARNING_RATE, type=float)
train_parser.add_argument('--dataloader-num-workers', default=DEFAULT_DATALOADER_NUM_WORKERS, type=int)
train_parser.add_argument('--model-lstm-hidden-size', default=DEFAULT_MODEL_LSTM_HIDDEN_SIZE, type=int)
train_parser.add_argument('--model-lstm-num-layers', default=DEFAULT_MODEL_LSTM_NUM_LAYERS, type=int)
train_parser.add_argument('--model-projection-size', default=DEFAULT_MODEL_PROJECTION_SIZE, type=int)
train_parser.add_argument('--random-seed', default=DEFAULT_RANDOM_SEED, type=int)
train_parser.add_argument('--restore-model-from', default='', type=str)
train_parser.add_argument('--restore-loss-from', default='', type=str)
# The following args won't be saved to config
train_parser.add_argument('--save-interval', default=DEFAULT_SAVING_INTERVAL, type=int)
train_parser.add_argument('--training-model-save-dir', default=DEFAULT_TRAINING_MODEL_DIR, type=Path)
train_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
train_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
train_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
train_parser.add_argument('-d', '--debug', action='store_true')

test_parser = main_subparser.add_parser('test')
test_parser.set_defaults(main_func=test_main)
test_parser.add_argument('model_dir', type=Path)
test_parser.add_argument('--N', type=int, default=10)
test_parser.add_argument('--M', type=int, default=6)
test_parser.add_argument('--epochs', type=int, default=300)
test_parser.add_argument('--dataloader-num-workers', type=int, default=1)
test_parser.add_argument('--random-seed', type=int)
test_parser.add_argument('--selected-iterations', type=str, nargs='*')
test_parser.add_argument('--stride', default=1, type=int)
test_parser.add_argument('--vox1test-mel-spectrogram-dir', default=DEFAULT_VOX1TEST_MEL_SPECTROGRAM_DIR, type=Path)
test_parser.add_argument('-d', '--debug', action='store_true')

nld_ge2e_centroids_parser = main_subparser.add_parser('nld-save-ge2e-embedding-centroid')
nld_ge2e_centroids_parser.set_defaults(main_func=nld_distance_main)
nld_ge2e_centroids_parser.add_argument('model_dir', type=Path)
nld_ge2e_centroids_parser.add_argument('--selected-iterations', type=str, nargs='*')
nld_ge2e_centroids_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
nld_ge2e_centroids_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
nld_ge2e_centroids_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
nld_ge2e_centroids_parser.add_argument('-d', '--debug', action='store_true')

nld_distance_parser = main_subparser.add_parser('nld-distance')
nld_distance_parser.set_defaults(main_func=nld_distance_main)
nld_distance_parser.add_argument('model_dir', type=Path)
nld_distance_parser.add_argument('--selected-iterations', type=str, nargs='*')
nld_distance_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
nld_distance_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
nld_distance_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
nld_distance_parser.add_argument('-d', '--debug', action='store_true')

nld_distance_parser = main_subparser.add_parser('nld-distance')
nld_distance_parser.set_defaults(main_func=nld_distance_main)
nld_distance_parser.add_argument('model_dir', type=Path)
nld_distance_parser.add_argument('--selected-iterations', type=str, nargs='*')
nld_distance_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
nld_distance_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
nld_distance_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
nld_distance_parser.add_argument('-d', '--debug', action='store_true')

nld_confidence_parser = main_subparser.add_parser('nld-confidence')
nld_confidence_parser.set_defaults(main_func=nld_confidence_main)
nld_confidence_parser.add_argument('model_dir', type=Path)
nld_confidence_parser.add_argument('--selected-iterations', type=str, nargs='*')
nld_confidence_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
nld_confidence_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
nld_confidence_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
nld_confidence_parser.add_argument('-d', '--debug', action='store_true')

nld_loss_parser = main_subparser.add_parser('nld-loss')
nld_loss_parser.set_defaults(main_func=nld_loss_main)
nld_loss_parser.add_argument('model_dir', type=Path)
nld_loss_parser.add_argument('--sampling-interval', type=int, default=5)
nld_loss_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
nld_loss_parser.add_argument('--vox1-mel-spectrogram-dir', default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR, type=Path)
nld_loss_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
nld_loss_parser.add_argument('-d', '--debug', action='store_true')

args = main_parser.parse_args()
args.main_func(args)
