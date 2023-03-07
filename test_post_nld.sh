SEED=$1 # 0, 1, 2, 3, 4
NLD_TYPE=$2 # "confidence" or "distance"

# Permute
python . test data/training-models/Permute-20-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Permute-20-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Permute-50-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Permute-50-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Permute-75-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Permute-75-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}

# Open
python . test data/training-models/Open-20-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Open-20-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Open-50-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Open-50-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Open-75-AAMSC-bs256-s15.0-m0.1-K10-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}
python . test data/training-models/Open-75-AAMSC-bs256-s15.0-m0.1-K3-seed${SEED} --selected-iterations final --use-nld-result ${NLD_TYPE}