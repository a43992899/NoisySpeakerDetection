#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

# clean
python . test data/training-models/clean-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/clean-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/clean-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/clean-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/clean-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/clean-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/clean-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/clean-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/clean-GE2E-bs128-M4-seed${SEED} --selected-iterations final
python . test data/training-models/clean-GE2E-bs256-M4-seed${SEED} --selected-iterations final

# permute
python . test data/training-models/Permute-20-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-GE2E-bs128-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-20-GE2E-bs256-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-GE2E-bs128-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-50-GE2E-bs256-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-GE2E-bs128-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-GE2E-bs256-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-GE2E-bs256-M8-seed${SEED} --selected-iterations final

# open-set
python . test data/training-models/Open-20-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-GE2E-bs128-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Open-20-GE2E-bs256-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-GE2E-bs128-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Open-50-GE2E-bs256-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAM-bs128-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAM-bs256-s15.0-m0.1-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAMSC-bs128-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAMSC-bs128-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-CE-bs128-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-CE-bs256-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-GE2E-bs128-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-GE2E-bs256-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-GE2E-bs256-M8-seed${SEED} --selected-iterations final
