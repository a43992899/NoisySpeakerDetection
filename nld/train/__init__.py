from pathlib import Path
from typing import List, Optional

from ..constant.config import TestConfig, TrainConfig
from .train import train, test


def train_main(args):
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir

    training_model_save_dir: Path = args.training_model_save_dir
    training_model_save_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        restore_model_from=args.restore_model_from,
        restore_loss_from=args.restore_loss_from,
        loss=args.loss,
        noise_level=args.noise_level,
        noise_type=args.noise_type,

        N=args.N,
        M=args.M,
        s=args.s,
        m=args.m,
        K=args.K,

        iterations=args.iterations,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.dataloader_num_workers,
        model_lstm_hidden_size=args.model_lstm_hidden_size,
        model_lstm_num_layers=args.model_lstm_num_layers,
        model_projection_size=args.model_projection_size,
        random_seed=args.random_seed,
    )
    debug: bool = args.debug
    save_interval: int = args.save_interval

    for mislabeled_json_file in mislabeled_json_dir.iterdir():
        if not mislabeled_json_file.is_file():
            continue
        mislabeled_json_file_name = mislabeled_json_file.stem
        if str(cfg.noise_level) in mislabeled_json_file_name and cfg.noise_type in mislabeled_json_file_name:
            break
    else:
        raise FileNotFoundError()

    train(
        cfg, mislabeled_json_file, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir,
        training_model_save_dir, save_interval, debug
    )


def test_main(args):
    model_dir: Path = args.model_dir
    selected_iterations: Optional[List[str]] = args.selected_iterations
    stride: int = args.stride
    vox1test_mel_spectrogram_dir: Path = args.vox1test_mel_spectrogram_dir
    debug: bool = args.debug

    if args.M % 2 != 0:
        raise ValueError()

    test_config = TestConfig(
        N=args.N,
        M=args.M,
        epochs=args.epochs,
        dataloader_num_workers=args.dataloader_num_workers,
        random_seed=args.random_seed
    )

    if selected_iterations is None:
        selected_iterations = sorted(map(
            lambda s: s.stem.split('-')[-1],
            filter(lambda p: p.stem.startswith('model'), model_dir.iterdir())
        ), reverse=True)
        selected_iterations = selected_iterations[0:len(selected_iterations) // 2:stride]

    test(test_config, model_dir, selected_iterations, vox1test_mel_spectrogram_dir, debug)
