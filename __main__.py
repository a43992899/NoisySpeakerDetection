from argparse import ArgumentParser
from pathlib import Path

from nld.constant.defaults import *
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
produce_noisy_label_parser.add_argument('--mislabeled-json-dir', type=Path,
                                        default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR)
produce_noisy_label_parser.add_argument('--vox1-output-dir', type=Path, default=DEFAULT_VOX1_MEL_SPECTROGRAM_DIR)
produce_noisy_label_parser.add_argument('--vox2-output-dir', type=Path, default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR)
produce_noisy_label_parser.add_argument('--random-seed', type=int, default=1)


train_parser = main_subparser.add_parser('train')
train_parser.set_defaults(main_func=train_main)
train_parser.add_argument('--enable-wandb', default=False, type=bool)
train_parser.add_argument('--mislabeled-json-dir', default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR, type=Path)
train_parser.add_argument('--vox2-mel-spectrogram-dir', default=DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, type=Path)
# train_parser.add_argument('--cfg', type=str, default='config/config.yaml', help='config.yaml path')
# train_parser.add_argument('--csv', type=str, default='../data/test_results.csv', help='csv path for writing test results')

args = main_parser.parse_args()
args.main_func(args)
