from pathlib import Path
from ..constant.config import NoiseLevel, NoiseType


def find_mislabeled_json(mislabeled_json_dir: Path, noise_type: NoiseType, noise_level: NoiseLevel):
    if noise_level == 0:
        return None

    for mislabeled_json_file in mislabeled_json_dir.iterdir():
        if mislabeled_json_file.suffix != '.json' or not mislabeled_json_file.is_file():
            continue
        mislabeled_json_file_name = mislabeled_json_file.stem
        if str(noise_level) in mislabeled_json_file_name and noise_type in mislabeled_json_file_name:
            break
    else:
        raise FileNotFoundError()

    return mislabeled_json_file
