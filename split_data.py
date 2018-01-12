import os
import random
import collections

import audioread

from pingu.utils import textfile

DATA_ROOT = '/cluster/home/buec/eva/data/'

all_path = os.path.join(DATA_ROOT, 'train.txt')
all_list = textfile.read_separated_lines(all_path, separator=' ')

cleaned_tracks = []

for track in all_list:
    track_id = track[0]

    with audioread.audio_open(track[1]) as f:
        length = f.duration

    if length > 29:
        cleaned_tracks.append(track)
    else:
        print('got rid of track {} because length {}'.format(track_id, length))

tracks_by_genre = collections.defaultdict(list)

for track in all_list:
    tracks_by_genre[track[2]].append(track)

train_tracks = []
dev_tracks = []

for genre, tracks in tracks_by_genre.items():
    train_n = int(len(tracks) * 0.8)
    train_tracks.extend(tracks[:train_n])
    dev_tracks.extend(tracks[train_n:])

train_tracks.sort(key=lambda x: x[0])
dev_tracks.sort(key=lambda x: x[0])

textfile.write_separated_lines(os.path.join(DATA_ROOT, 'tr_split.txt'), train_tracks, separator=' ')
textfile.write_separated_lines(os.path.join(DATA_ROOT, 'dev_split.txt'), dev_tracks, separator=' ')
