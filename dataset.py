import numpy as np


class RawDataset(object):
    def __init__(self, track_list, samples):
        self.track_list = track_list
        self.samples = samples

    def __len__(self):
        return len(self.track_list) * 116

    def __getitem__(self, item):
        track_index = int(item / 116)
        frame_index = item - (track_index * 116)

        track = self.track_list[track_index]

        label = np.zeros(16).astype(np.float32)
        label[int(track[2])] = 1

        feats = self.samples.get(str(track[0]))[()].astype(np.float32)
        start = frame_index * 11025
        feats = feats[start:(start + 44100)]

        if feats.size < 44100:
            feats = np.pad(feats, (0, 44100 - feats.shape[0]), mode='constant', constant_values=0)

        return feats, label