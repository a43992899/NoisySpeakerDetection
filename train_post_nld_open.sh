#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4
NLD_TYPE=$2 # "confidence" or "distance"

# OpenSet Noise
python . train AAMSC 20 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}
python . train AAMSC 50 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}
python . train AAMSC 75 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}
python . train AAMSC 20 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}
python . train AAMSC 50 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}
python . train AAMSC 75 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED} --use-nld-result ${NLD_TYPE}

