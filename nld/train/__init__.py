from pathlib import Path
from typing import List, Optional

from nld.process_data.mislabel import find_mislabeled_json

from ..constant.config import TestConfig, TrainConfig
from .train import test, train


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

    mislabeled_json_file = find_mislabeled_json(mislabeled_json_dir, cfg.noise_type, cfg.noise_level)

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

    if selected_iterations is None:
        selected_iterations = sorted(map(
            lambda s: s.stem.split('-')[-1],
            filter(lambda p: p.stem.startswith('model'), model_dir.iterdir())
        ), reverse=True)
        selected_iterations = selected_iterations[0:len(selected_iterations) // 2:stride]

    if args.M % 2 != 0:
        raise ValueError()

    for selected_iteration in selected_iterations:
        test_config = TestConfig(
            N=args.N,
            M=args.M,
            epochs=args.epochs,
            dataloader_num_workers=args.dataloader_num_workers,
            random_seed=args.random_seed,
            model_dir=model_dir,
            iteration=selected_iteration
        )

        test(test_config, vox1test_mel_spectrogram_dir, debug)
