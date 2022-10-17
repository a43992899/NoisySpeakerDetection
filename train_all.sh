#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

bash train_clean.sh ${SEED}
bash train_permute.sh ${SEED}
bash train_open.sh ${SEED}
