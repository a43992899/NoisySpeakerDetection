from argparse import ArgumentParser
from pathlib import Path

from nld.constant.defaults import (DEFAULT_VOX1_AUDIO_DIR,
                                   DEFAULT_VOX1_OUTPUT_DIR,
                                   DEFAULT_VOX1TEST_AUDIO_DIR,
                                   DEFAULT_VOX1TEST_OUTPUT_DIR,
                                   DEFAULT_VOX2_AUDIO_DIR,
                                   DEFAULT_VOX2_OUTPUT_DIR,
                                   DEFAULT_VOXCELEB_MISLABELED_JSON_DIR)
from nld.process_data import produce_mel_spectrogram, produce_noisy_label
# from nld.train_speech_embedder import main as train_main

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
produce_mel_spectrogram_parser.add_argument('--vox1-output-dir', type=Path, default=DEFAULT_VOX1_OUTPUT_DIR)
produce_mel_spectrogram_parser.add_argument('--vox1test-output-dir', type=Path, default=DEFAULT_VOX1TEST_OUTPUT_DIR)
produce_mel_spectrogram_parser.add_argument('--vox2-output-dir', type=Path, default=DEFAULT_VOX2_OUTPUT_DIR)
produce_mel_spectrogram_parser.add_argument('--sr', type=int, default=16000)
produce_mel_spectrogram_parser.add_argument('--nfft', type=int, default=512)
produce_mel_spectrogram_parser.add_argument('--window', type=float, default=0.025)
produce_mel_spectrogram_parser.add_argument('--hop', type=float, default=0.01)
produce_mel_spectrogram_parser.add_argument('--nmels', type=int, default=40)
produce_mel_spectrogram_parser.add_argument('--tisv-frame', type=int, default=180)
produce_mel_spectrogram_parser.add_argument('--silence-threshold', type=int, default=30)

produce_noisy_label_parser = main_subparser.add_parser('produce-noisy-label')
produce_noisy_label_parser.set_defaults(main_func=produce_noisy_label)
produce_noisy_label_parser.add_argument('--mislabeled-json-dir', type=Path,
                                        default=DEFAULT_VOXCELEB_MISLABELED_JSON_DIR)
produce_noisy_label_parser.add_argument('--vox1-output-dir', type=Path, default=DEFAULT_VOX1_OUTPUT_DIR)
produce_noisy_label_parser.add_argument('--vox2-output-dir', type=Path, default=DEFAULT_VOX2_OUTPUT_DIR)


# train_parser = main_subparser.add_parser('train')
# train_parser.set_defaults(main_func=train_main)
# train_parser.add_argument('--cfg', type=str, default='config/config.yaml', help='config.yaml path')
# train_parser.add_argument('--csv', type=str, default='../data/test_results.csv', help='csv path for writing test results')

args = main_parser.parse_args()
args.main_func(args)
