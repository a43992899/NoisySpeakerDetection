import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb
from torch import Tensor
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as cuda_is_available
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..constant.config import DataConfig, TrainConfig
from ..constant.entities import WANDB_ENTITY, WANDB_TRAINING_PROJECT_NAME
from ..model.loss import AAMSoftmax, GE2ELoss, SubcenterArcMarginProduct
from ..model.model import SpeechEmbedder
from ..process_data.dataset import SpeakerDataset
from ..utils import current_utc_time, set_random_seed_to


def train(
    cfg: TrainConfig, mislabeled_json_file: Path, vox1_mel_spectrogram_dir: Path,
    vox2_mel_spectrogram_dir: Path, training_model_save_dir: Path, save_interval: int, debug: bool
):
    set_random_seed_to(cfg.random_seed)
    job_name = f'{cfg.description}-{current_utc_time()}'
    device = torch.device('cuda' if cuda_is_available() else 'cpu')

    if not debug:
        wandb.init(
            project=WANDB_TRAINING_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=cfg.to_dict(),
            name=job_name
        )

    checkpoint_dir = training_model_save_dir / job_name
    if not debug:
        checkpoint_dir.mkdir()
        cfg.to_json(checkpoint_dir / 'config.json')
        print(f'Checkpoints will be saved to `{checkpoint_dir}`')

    spkr2id_file = vox2_mel_spectrogram_dir / 'speaker-label-to-id.json'
    with open(spkr2id_file, 'r') as f:
        speaker_label_to_id: Dict[str, int] = json.load(f)
    utterance_classes_num = len(speaker_label_to_id)

    data_processing_config = DataConfig.from_json(vox2_mel_spectrogram_dir / 'data-processing-config.json')

    embedder_net = SpeechEmbedder(
        data_processing_config.nmels,
        cfg.model_lstm_hidden_size,
        cfg.model_lstm_num_layers,
        cfg.model_projection_size,
        should_softmax=(cfg.loss == 'CE'),
        num_classes=utterance_classes_num
    ).to(device)

    if cfg.loss == 'CE':
        cfg.assert_attr('N', 'M')
        criterion = torch.nn.NLLLoss()
    elif cfg.loss == 'GE2E':
        cfg.assert_attr('N', 'M')
        criterion = GE2ELoss()
    elif cfg.loss == 'AAM':
        cfg.assert_attr('N', 'M', 's', 'm')
        print(f'At an early stage, easy_margin is enabled for AAM. '
              f'easy_margin flag will be turned down after {cfg.iterations // 8} iterations.')
        criterion = AAMSoftmax(
            cfg.model_projection_size, utterance_classes_num,
            scale=cfg.s, margin=cfg.m, easy_margin=True
        )
    else:
        assert cfg.loss == 'AAMSC'
        cfg.assert_attr('N', 'M', 's', 'm', 'K')
        print(f'At an early stage, easy_margin is enabled for AAMSC. '
              f'easy_margin flag will be turned down after {cfg.iterations // 8} iterations.')
        criterion = SubcenterArcMarginProduct(
            cfg.model_projection_size, utterance_classes_num,
            s=cfg.s, m=cfg.m, K=cfg.K, easy_margin=True,
        )
    criterion.to(device)

    train_dataset = SpeakerDataset(cfg.M, vox1_mel_spectrogram_dir, vox2_mel_spectrogram_dir, mislabeled_json_file)
    train_data_loader = DataLoader(
        train_dataset,
        # Note that the real batch size = N * M
        batch_size=cfg.N,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': embedder_net.parameters()},
            {'params': criterion.parameters()}
        ], lr=cfg.learning_rate)
    else:
        raise ValueError(f'Unsupported module optimizer {cfg.optimizer}.')

    scaler = GradScaler()

    if (restore_model := cfg.restore_model_from) and (restore_loss := cfg.restore_loss_from):
        restore_model = Path(restore_model)
        restore_loss = Path(restore_loss)
        missing_keys, unexpected_keys = embedder_net.load_state_dict(torch.load(restore_model))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()
        missing_keys, unexpected_keys = criterion.load_state_dict(torch.load(restore_loss))
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            raise ValueError()

    gc.collect()
    if cuda_is_available():
        empty_cuda_cache()

    if not debug:
        wandb.watch(embedder_net, criterion)

    embedder_net.train()
    total_iterations = 0
    with tqdm(total=cfg.iterations) as progress_bar:
        while total_iterations < cfg.iterations:
            total_loss = 0.0
            local_iterations = 0

            for mels, is_noisy, y, _, _ in train_data_loader:
                mels: Tensor = mels.to(device, non_blocking=True)
                is_noisy: Tensor = is_noisy.to(device)
                y: Tensor = y.to(device)

                assert mels.dim() == 4
                mels = mels.reshape((cfg.N * cfg.M, mels.size(2), mels.size(3)))

                optimizer.zero_grad()

                with autocast():
                    embeddings = embedder_net(mels)

                    if cfg.loss == 'GE2E':
                        criterion: GE2ELoss
                        embeddings = embeddings.reshape((cfg.N, cfg.M, embeddings.size(1)))
                        loss, _ = criterion(embeddings, y)
                    else:
                        assert cfg.loss in ('AAM', 'AAMSC', 'CE')
                        if cfg.loss != 'CE' and total_iterations == cfg.iterations // 8:
                            criterion.easy_margin = False
                        loss = criterion(embeddings, y.flatten())
                        if isinstance(loss, tuple):
                            loss, _ = loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss

                progress_bar.update(1)
                total_iterations += 1
                local_iterations += 1
                if total_iterations >= cfg.iterations:
                    break
                if (not debug) and total_iterations % save_interval == 0:
                    iteration_string = str(total_iterations).zfill(len(str(cfg.iterations)))
                    torch.save(embedder_net.state_dict(), checkpoint_dir / f'model-{iteration_string}.pth')
                    torch.save(criterion.state_dict(), checkpoint_dir / f'loss-{iteration_string}.pth')

            if not debug:
                wandb.log({'Loss': total_loss / local_iterations})

    if not debug:
        torch.save(embedder_net.state_dict(), checkpoint_dir / 'model-final.pth')
        torch.save(criterion.state_dict(), checkpoint_dir / 'loss-final.pth')
        wandb.finish()


# def test_one(hp: Config):
#     model_path = hp.test.model_path
#     print(model_path)
#     device = torch.device(hp.device)
#     random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.manual_seed_all(0)

#     test_dataset = SpeakerDatasetPreprocessed(hp)

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=hp.test.N,
#         shuffle=True,
#         num_workers=hp.test.num_workers,
#         drop_last=True)

#     try:
#         embedder_net = SpeechEmbedder(hp).to(device)
#         embedder_net.load_state_dict(torch.load(model_path))
#     except BaseException:
#         embedder_net = SpeechEmbedder_Softmax(
#             hp=hp, num_classes=5994).to(device)
#         embedder_net.load_state_dict(torch.load(model_path))
#     embedder_net.eval()

#     model_parameters = filter(lambda p: p.requires_grad, embedder_net.parameters())
#     num_params = sum([np.prod(p.size()) for p in model_parameters])
#     print("Number of params: ", num_params)

#     ypreds = []
#     ylabels = []

#     for _ in range(hp.test.epochs):
#         for batch_id, (mel_db_batch, labels, is_noisy, utterance_ids) in enumerate(tqdm(test_loader)):
#             assert hp.test.M % 2 == 0

#             utterance_ids = np.array(utterance_ids).T
#             mel_db_batch = mel_db_batch.to(device)
#             enrollment_batch, verification_batch = torch.split(
#                 mel_db_batch, int(mel_db_batch.size(1) / 2), dim=1)
#             enrollment_batch = torch.reshape(
#                 enrollment_batch,
#                 (hp.test.N * hp.test.M // 2,
#                  enrollment_batch.size(2),
#                  enrollment_batch.size(3)))
#             verification_batch = torch.reshape(
#                 verification_batch,
#                 (hp.test.N * hp.test.M // 2,
#                  verification_batch.size(2),
#                  verification_batch.size(3)))

#             perm = torch.randperm(verification_batch.size(0))
#             unperm = torch.argsort(perm)

#             verification_batch = verification_batch[perm]
#             # get embedder_net attribute
#             if embedder_net.__class__.__name__ == 'SpeechEmbedder_Softmax':
#                 enrollment_embeddings = embedder_net.get_embedding(
#                     enrollment_batch)
#                 verification_embeddings = embedder_net.get_embedding(
#                     verification_batch)
#             else:
#                 enrollment_embeddings = embedder_net(enrollment_batch)
#                 verification_embeddings = embedder_net(verification_batch)

#             verification_embeddings = verification_embeddings[unperm]

#             enrollment_embeddings = torch.reshape(
#                 enrollment_embeddings,
#                 (hp.test.N,
#                  hp.test.M // 2,
#                  enrollment_embeddings.size(1)))
#             verification_embeddings = torch.reshape(
#                 verification_embeddings,
#                 (hp.test.N,
#                  hp.test.M // 2,
#                  verification_embeddings.size(1)))

#             enrollment_centroids = get_centroids(enrollment_embeddings)
#             veri_embed = torch.cat(
#                 [verification_embeddings[:, 0], verification_embeddings[:, 1], verification_embeddings[:, 2]])

#             veri_embed_norm = veri_embed / \
#                 torch.norm(veri_embed, dim=1).unsqueeze(-1)
#             enrl_embed = torch.cat([enrollment_centroids] * 1)
#             enrl_embed_norm = enrl_embed / \
#                 torch.norm(enrl_embed, dim=1).unsqueeze(-1)
#             sim_mat = torch.matmul(
#                 veri_embed_norm, enrl_embed_norm.transpose(-1, -2)).data.cpu().numpy()
#             truth = np.ones_like(sim_mat) * (-1)
#             for i in range(truth.shape[0]):
#                 truth[i, i % 10] = 1
#             ypreds.append(sim_mat.flatten())
#             ylabels.append(truth.flatten())

#         eer, thresh = compute_eer(ypreds, ylabels)
#         print("eer:", eer, "threshold:", thresh)
#     print(model_path, 'eval done.')
#     return eer, thresh


def test(model_dir: Path, selected_iterations: Optional[List[int]], vox1_mel_spectrogram_dir: Path):
    config = TrainConfig.from_json(model_dir / 'config.json')

    if selected_iterations is None:
        pass
    else:
        pass


    if os.path.isfile(hp.test.model_path):
        test_one(hp)
    elif os.path.isdir(hp.test.model_path):
        if not os.path.exists(csv_path):
            csv_header_line = 'ModelPath,EER(%),Threshold(%)\n'
            write_to_csv(csv_path, csv_header_line)
            tested_model_paths = []
        else:
            # read csv, get the model paths
            csv_file = open(csv_path, 'r')
            csv_lines = csv_file.readlines()
            csv_file.close()
            tested_model_paths = [line.split(',')[0] for line in csv_lines[1:]]

        pth_list = sorted(get_all_file_with_ext(hp.test.model_path, '.pth'))
        for file in pth_list:
            if 'ckpt_criterion_epoch' in file or file in tested_model_paths:
                continue
            else:
                file_to_test = None
                if '/GE2E/' in file:
                    if isTarget(
                        file,
                        target_strings=[
                            'ckpt_epoch_100.pth',
                            'ckpt_epoch_200.pth',
                            'ckpt_epoch_300.pth',
                            'ckpt_epoch_400.pth',
                            'ckpt_epoch_800.pth']):
                        file_to_test = file
                    else:
                        continue
                elif isTarget(file, target_strings=['/Softmax/', '/AAM/', '/AAMSC/']):
                    if 'bs128' in file and 'ckpt_epoch_1600.pth' in file:
                        file_to_test = file
                    elif 'bs256' in file and 'ckpt_epoch_3200.pth' in file:
                        file_to_test = file
                    else:
                        continue
                else:
                    continue
                eer, thresh = test_one(file_to_test)
                csv_line = f"{file_to_test},{eer*100},{thresh*100}\n"
                write_to_csv(csv_path, csv_line)

    else:
        print("model_path is not a file or a directory")
