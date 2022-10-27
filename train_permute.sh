#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

# # Permute Noise
# python . train CE 20 Permute --N 128 --M 1 --random-seed ${SEED}
# python . train CE 20 Permute --N 256 --M 1 --random-seed ${SEED}
# python . train AAM 20 Permute --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
# python . train AAM 20 Permute --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 20 Permute --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 20 Permute --N 16 --M 8 --random-seed ${SEED}
# python . train AAMSC 20 Permute --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 20 Permute --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
# python . train AAMSC 20 Permute --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 20 Permute --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
# python . train CE 50 Permute --N 128 --M 1 --random-seed ${SEED}
# python . train CE 50 Permute --N 256 --M 1 --random-seed ${SEED}
# python . train AAM 50 Permute --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
# python . train AAM 50 Permute --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 50 Permute --N 32 --M 8 --random-seed ${SEED}
python . train GE2E 50 Permute --N 16 --M 8 --random-seed ${SEED}
# python . train AAMSC 50 Permute --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 50 Permute --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
# python . train AAMSC 50 Permute --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 50 Permute --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
# python . train CE 75 Permute --N 128 --M 1 --random-seed ${SEED}
# python . train CE 75 Permute --N 256 --M 1 --random-seed ${SEED}
# python . train AAM 75 Permute --N 128 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
# python . train AAM 75 Permute --N 256 --M 1 --s 15 --m 0.1 --random-seed ${SEED}
python . train GE2E 75 Permute --N 16 --M 16 --random-seed ${SEED}
python . train GE2E 75 Permute --N 8 --M 16 --random-seed ${SEED}
# python . train AAMSC 75 Permute --N 128 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 75 Permute --N 128 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
# python . train AAMSC 75 Permute --N 256 --M 1 --s 15 --m 0.1 --K 3 --random-seed ${SEED}
# python . train AAMSC 75 Permute --N 256 --M 1 --s 15 --m 0.1 --K 10 --random-seed ${SEED}
