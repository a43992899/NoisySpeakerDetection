#!/bin/bash
SEED=$1 # 0, 1, 2, 3, 4

python . test data/training-models/Open-75-GE2E-bs256-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-GE2E-bs256-M16-seed${SEED} --selected-iterations final
python . test data/training-models/Open-75-GE2E-bs256-M8-seed${SEED} --selected-iterations final
python . test data/training-models/Permute-75-GE2E-bs256-M8-seed${SEED} --selected-iterations final
