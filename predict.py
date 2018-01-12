import os
import logging
import argparse

import h5py
import numpy as np

import torch
from torch import autograd

from pingu.utils import textfile
from pingu.utils import jsonfile

import model

DATA_ROOT = '/cluster/home/buec/eva/data'
OUT_PATH = '/cluster/home/buec/eva/output'

TRACK_LENGTH = 30 * 44100

parser = argparse.ArgumentParser()

parser.add_argument('out_folder', type=str)
parser.add_argument('model_state', type=str)
parser.add_argument('--no_cuda', action="store_true")

args = parser.parse_args()

#
#
#

out_folder = os.path.join(OUT_PATH, args.out_folder)
os.makedirs(out_folder, exist_ok=True)

logging.getLogger().handlers = []
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

fh = logging.FileHandler(os.path.join(out_folder, 'log.txt'))
fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
logging.getLogger().addHandler(fh)

#
#
#

test_path = os.path.join(DATA_ROOT, 'test.txt')
test_list = textfile.read_separated_lines(test_path, separator=' ')
test_samples_path = os.path.join(DATA_ROOT, 'test_samples.h5')
samples = h5py.File(test_samples_path, 'r')

#
#
#

model = model.ModelRaw()
model.load_state_dict(torch.load(os.path.join(OUT_PATH, args.model_state)))

model.eval()

if not args.no_cuda:
    model.cuda()

#
#
#

classes = jsonfile.read_json_file(os.path.join(DATA_ROOT, 'classes.json'))

f = open(os.path.join(out_folder, 'result.csv'), 'w')
f.write('file_id,{}\n'.format(','.join(classes)))

for track in test_list:
    track_id = track[0]
    track_data = samples.get(track_id)[()].astype(np.float32)
    track_data = track_data[:TRACK_LENGTH]

    if track_data.size < TRACK_LENGTH:
        track_data = np.pad(track_data, (0, TRACK_LENGTH - track_data.size), mode='constant', constant_values=0)

    track_data = track_data.reshape(-1, 44100)
    input_var = autograd.Variable(torch.from_numpy(track_data))

    if not args.no_cuda:
        input_var = input_var.cuda()

    prediction = model.forward(input_var).data.cpu().numpy()
    avg = prediction.mean(axis=0)

    f.write('{},{}\n'.format(track_id, ','.join(str(x) for x in avg)))

f.close()
