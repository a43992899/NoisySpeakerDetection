from nld.nld import compute_and_save_ge2e_embedding_centroid
from nld.constant.defaults import *


compute_and_save_ge2e_embedding_centroid(
    DEFAULT_TRAINING_MODEL_DIR, 'final', DEFAULT_VOX1_MEL_SPECTROGRAM_DIR,
    DEFAULT_VOX2_MEL_SPECTROGRAM_DIR, DEFAULT_VOXCELEB_MISLABELED_JSON_DIR
)
