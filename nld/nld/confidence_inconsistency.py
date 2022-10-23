import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot, log_softmax
from torch.cuda import is_available as cuda_is_available

from ..model.loss import AAMSoftmax, SubcenterArcMarginProduct

from ..constant.config import DataConfig, TrainConfig
from ..process_data.dataset import VOX2_CLASS_NUM, SpeakerLabelDataset
from ..process_data.mislabel import find_mislabeled_json


@torch.no_grad()
def confidence_inconsistency_evaluation(
    model_dir: Path, selected_iteration: str, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, mislabeled_json_dir: Path, debug: bool,
):
    device = torch.device('cuda' if cuda_is_available() else 'cpu')
    train_config = TrainConfig.from_json(model_dir / 'config.json')
    data_processing_config = DataConfig.from_json(
        vox2_mel_spectrogram_dir / 'data-processing-config.json'
    )
    mislabeled_json_file = find_mislabeled_json(
        mislabeled_json_dir, train_config.noise_type, train_config.noise_level
    )

    model = train_config.forge_model(
        data_processing_config.nmels, VOX2_CLASS_NUM,
    ).to(device).eval()
    model.load_state_dict(torch.load(
        model_dir / f'model-{selected_iteration}.pth', map_location=device,
    ))
    with open(vox2_mel_spectrogram_dir / 'speaker-label-to-id.json', 'r') as f:
        utterance_classes_num = len(json.load(f))
    criterion = train_config.forge_criterion(utterance_classes_num).to(device).eval()
    criterion.load_state_dict(torch.load(
        model_dir / f'loss-{selected_iteration}.pth', map_location=device,
    ))

    dataset = SpeakerLabelDataset(
        vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file
    )

    if train_config.loss == 'GE2E':
        pass
    else:
        for i in range(len(dataset)):
            mel, is_noisy, label = dataset[i]
            mel: Tensor = mel.to(device)
            y = one_hot(torch.tensor(label), VOX2_CLASS_NUM)
            model_output: Tensor = model(mel)
            if train_config.loss == 'CE':
                inconsistencies = np.fromiter(
                    (torch.max(y * model_output[j, ...]).item() for j in range(model_output.size(0))),
                    dtype=np.float32,
                )
            else:
                assert train_config.loss in ('AAM', 'AAMSC')
                assert isinstance(criterion, (AAMSoftmax, SubcenterArcMarginProduct))
                cosine = criterion.directly_predict(model_output)
                inconsistencies = np.fromiter(
                    (torch.max(y * cosine[j, ...]).item() for j in range(model_output.size(0))),
                    dtype=np.float32,
                )

    # TODO: do the following:
    # 1. CE do model forward, else do model get embedding and criterion predict
    # 2. call BMM and evaluation.
