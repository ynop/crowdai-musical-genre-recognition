import os

import numpy as np
import h5py
import librosa

from pingu.utils import textfile

DATA_ROOT = '/cluster/home/buec/eva/data'


def extract_features(track_list, out):
    f = h5py.File(out, 'a')

    for index, info in enumerate(track_list):
        print(index)

        track_id = info[0]
        track_path = info[1]

        samples, sampling_rate = librosa.core.load(track_path, sr=None, mono=True)
        feats = samples.astype(np.float32)
        f.create_dataset(str(track_id), data=feats)

    f.close()


train_path = os.path.join(DATA_ROOT, 'train.txt')
test_path = os.path.join(DATA_ROOT, 'test.txt')

train_list = textfile.read_separated_lines(train_path, separator=' ')
test_list = textfile.read_separated_lines(test_path, separator=' ')

print('TRAIN')
extract_features(train_list, os.path.join(DATA_ROOT, 'train_samples.h5'))

print('TEST')
extract_features(test_list, os.path.join(DATA_ROOT, 'test_samples.h5'))
