from pathlib import Path
from ..constant.config import NoiseLevel, NoiseType


def find_mislabeled_json(mislabeled_json_dir: Path, noise_type: NoiseType, noise_level: NoiseLevel):
    if noise_level == 0:
        return None

    for mislabeled_json_file in mislabeled_json_dir.iterdir():
        if mislabeled_json_file.suffix != '.json' or not mislabeled_json_file.is_file():
            continue
        if str(noise_level) in mislabeled_json_file.stem and noise_type in mislabeled_json_file.stem:
            return mislabeled_json_file

    raise FileNotFoundError()
