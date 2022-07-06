#!/bin/bash

/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/Softmax/20%/bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/Softmax/50%/bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/Softmax/75%/bs256/log/config.yaml

/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/GE2E/20%/m8_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/GE2E/50%/m8_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/GE2E/75%/m16_bs256/log/config.yaml

/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAMSC/20%/m0.1_s15_k10_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAMSC/50%/m0.1_s15_k10_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAMSC/75%/m0.1_s15_k10_bs256/log/config.yaml

/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAM/20%/m0.1_s15_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAM/50%/m0.1_s15_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Permute/AAM/75%/m0.1_s15_bs256/log/config.yaml
