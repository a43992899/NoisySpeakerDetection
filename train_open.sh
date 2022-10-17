#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

# OpenSet Noise
python . train CE 20 Open --N 128 --M 1 --random-seed ${SEED}
python . train CE 20 Open --N 256 --M 1 --random-seed ${SEED}
python . train AAM 20 Open --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train AAM 20 Open --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 20 Open --N 64 --M 4 --random-seed ${SEED}
python . train GE2E 20 Open --N 32 --M 4 --random-seed ${SEED}
python . train GE2E 20 Open --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 20 Open --N 16 --M 8 --random-seed ${SEED}
python . train GE2E 20 Open --N 16 --M 16 --random-seed ${SEED}
python . train GE2E 20 Open --N 8 --M 16 --random-seed ${SEED}
python . train AAMSC 20 Open --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 20 Open --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train AAMSC 20 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 20 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train CE 50 Open --N 128 --M 1 --random-seed ${SEED}
python . train CE 50 Open --N 256 --M 1 --random-seed ${SEED}
python . train AAM 50 Open --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train AAM 50 Open --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 50 Open --N 64 --M 4 --random-seed ${SEED}
python . train GE2E 50 Open --N 32 --M 4 --random-seed ${SEED}
python . train GE2E 50 Open --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 50 Open --N 16 --M 8 --random-seed ${SEED}
python . train GE2E 50 Open --N 16 --M 16 --random-seed ${SEED}
python . train GE2E 50 Open --N 8 --M 16 --random-seed ${SEED}
python . train AAMSC 50 Open --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 50 Open --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train AAMSC 50 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 50 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train CE 75 Open --N 128 --M 1 --random-seed ${SEED}
python . train CE 75 Open --N 256 --M 1 --random-seed ${SEED}
python . train AAM 75 Open --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train AAM 75 Open --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 75 Open --N 64 --M 4 --random-seed ${SEED}
python . train GE2E 75 Open --N 32 --M 4 --random-seed ${SEED}
python . train GE2E 75 Open --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 75 Open --N 16 --M 8 --random-seed ${SEED}
python . train GE2E 75 Open --N 16 --M 16 --random-seed ${SEED}
python . train GE2E 75 Open --N 8 --M 16 --random-seed ${SEED}
python . train AAMSC 75 Open --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 75 Open --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
python . train AAMSC 75 Open --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
python . train AAMSC 75 Open --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
