#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

bash run_clean_exp.sh ${SEED}
bash run_permute_exp.sh ${SEED}
bash run_open_exp.sh ${SEED}
