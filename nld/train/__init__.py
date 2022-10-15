from pathlib import Path

from ..constant.config import NewTrainConfig
from .train import train


def train_main(args):
    mislabeled_json_dir: Path = args.mislabeled_json_dir
    if not mislabeled_json_dir.exists():
        raise FileNotFoundError()
    if not mislabeled_json_dir.is_dir():
        raise NotADirectoryError()
    vox1_mel_spectrogram_dir: Path = args.vox1_mel_spectrogram_dir
    if not vox1_mel_spectrogram_dir.exists():
        raise FileNotFoundError()
    if not vox1_mel_spectrogram_dir.is_dir():
        raise NotADirectoryError()
    vox2_mel_spectrogram_dir: Path = args.vox2_mel_spectrogram_dir
    if not vox2_mel_spectrogram_dir.exists():
        raise FileNotFoundError()
    if not vox2_mel_spectrogram_dir.is_dir():
        raise NotADirectoryError()

    training_model_save_dir: Path = args.training_model_save_dir
    training_model_save_dir.mkdir(parents=True, exist_ok=True)

    cfg = NewTrainConfig(
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

        iterations=args.epoches,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.dataloader_num_workers,
        model_lstm_hidden_size=args.model_lstm_hidden_size,
        model_lstm_num_layers=args.model_lstm_num_layers,
        model_projection_size=args.model_projection_size,
        random_seed=args.random_seed,
    )
    enable_wandb: bool = args.enable_wandb

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
        training_model_save_dir, enable_wandb
    )


def test_main(args):
    pass
