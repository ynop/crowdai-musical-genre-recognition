import os
import glob

from pingu.utils import jsonfile
from pingu.utils import textfile

import fma

DATA_ROOT = '/cluster/home/buec/eva/data/'


def create_test_list():
    path = os.path.join(DATA_ROOT, 'crowdai_fma_test')
    out = os.path.join(DATA_ROOT, 'test.txt')

    mp3s = glob.glob(os.path.join(path, '*.mp3'))
    map = {os.path.splitext((os.path.basename(x)))[0]: x for x in mp3s}

    textfile.write_separated_lines(out, map, separator=' ', sort_by_column=0)


def create_train_list():
    out = os.path.join(DATA_ROOT, 'test.txt')

    tracks = fma.load(os.path.join(DATA_ROOT, 'fma_metadata', 'tracks.csv'))
    subset = tracks.index[tracks['set', 'subset'] <= 'medium']
    labels = tracks.loc[subset, ('track', 'genre_top')]

    classes = sorted(labels.unique())
    jsonfile.write_json_to_file(os.path.join(DATA_ROOT, 'classes.json'), classes)

    m = {v: i for i, v in enumerate(classes)}
    labels = labels.map(m).values

    map = {}

    for i in range(subset.size):
        track_id = subset[i]
        tid_str = '{:06d}'.format(track_id)
        track_path = os.path.join(DATA_ROOT, 'fma_medium', tid_str[:3], tid_str + '.mp3')

        map[track_id] = (track_path, labels[i])

    textfile.write_separated_lines(out, map, separator=' ', sort_by_column=0)


create_test_list()
create_train_list()