import os
import random
from data_load_test import SpeakerDatasetPreprocessedTest

mypath = 'data/voxceleb/vox2/spmel/'
filenames = [mypath + f for f in os.listdir(mypath)]
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames)
split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]



train_dataset = SpeakerDatasetPreprocessedTest(train_filenames)