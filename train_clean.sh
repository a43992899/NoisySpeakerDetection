#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

# noise level = 0 corresponding to clean
python . train CE 0 Permute --N 128 --M 1 --random-seed ${SEED}
python . train CE 0 Permute --N 256 --M 1 --random-seed ${SEED}
python . train AAM 0 Permute --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train AAM 0 Permute --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 0 Permute --N 64 --M 4 --random-seed ${SEED}
python . train GE2E 0 Permute --N 32 --M 4 --random-seed ${SEED}
python . train GE2E 0 Permute --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 0 Permute --N 16 --M 8 --random-seed ${SEED}
python . train GE2E 0 Permute --N 16 --M 16 --random-seed ${SEED}
python . train GE2E 0 Permute --N 8 --M 16 --random-seed ${SEED}
python . train AAMSC 0 Permute --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 0 Permute --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train AAMSC 0 Permute --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 0 Permute --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
