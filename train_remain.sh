#!/bin/bash

/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Open/AAMSC/20%/m0.1_s15_k3_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Open/AAMSC/50%/m0.1_s15_k3_bs256/log/config.yaml
/home/yrb/miniconda3/envs/speechbrain_ENV/bin/python /home/yrb/code/speechbrain/core/train_speech_embedder.py --cfg /home/yrb/code/speechbrain/data/models/Open/AAMSC/75%/m0.1_s15_k3_bs256/log/config.yaml
