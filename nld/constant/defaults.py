from pathlib import Path

DEFAULT_DATA_DIR = Path('./data')

DEFAULT_VOXCELEB_DATASET_DIR = DEFAULT_DATA_DIR / 'voxceleb'
DEFAULT_VOX1_AUDIO_DIR = DEFAULT_VOXCELEB_DATASET_DIR / 'vox1' / 'wav'
DEFAULT_VOX1TEST_AUDIO_DIR = DEFAULT_VOXCELEB_DATASET_DIR / 'vox1_test' / 'wav'
DEFAULT_VOX2_AUDIO_DIR = DEFAULT_VOXCELEB_DATASET_DIR / 'vox2' / 'wav'

DEFAULT_VOXCELEB_MEL_SPECTROGRAM_DIR = DEFAULT_DATA_DIR / 'voxceleb-mel'
DEFAULT_VOX1_MEL_SPECTROGRAM_DIR = DEFAULT_VOXCELEB_MEL_SPECTROGRAM_DIR / 'vox1'
DEFAULT_VOX1TEST_MEL_SPECTROGRAM_DIR = DEFAULT_VOXCELEB_MEL_SPECTROGRAM_DIR / 'vox1_test'
DEFAULT_VOX2_MEL_SPECTROGRAM_DIR = DEFAULT_VOXCELEB_MEL_SPECTROGRAM_DIR / 'vox2'

DEFAULT_VOXCELEB_MISLABELED_JSON_DIR = DEFAULT_DATA_DIR / 'voxceleb-mislabeled-json'

DEFAULT_TRAINING_MODEL_DIR = DEFAULT_DATA_DIR / 'training-models'

DEFAULT_RANDOM_SEED = 1

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NFFT = 512
DEFAULT_WINDOW_LENGTH = 0.025
DEFAULT_HOP_SIZE = 0.01
DEFAULT_NMELS = 40
DEFAULT_TISV_FRAME = 180
DEFAULT_SILENCE_THRESHOLD = 30

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_DATALOADER_NUM_WORKERS = 6
DEFAULT_EPOCHES = 3200

DEFAULT_MODEL_LSTM_HIDDEN_SIZE = 768
DEFAULT_MODEL_LSTM_NUM_LAYERS = 3
DEFAULT_MODEL_PROJECTION_SIZE = 256
