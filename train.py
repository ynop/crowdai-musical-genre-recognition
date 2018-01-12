import os
import logging
import argparse
import collections
import random

import h5py

import torch
from torch import optim
from torch.utils import data

import candle
from candle import metrics
from candle import callbacks

from pingu.utils import textfile

import model
import utils
import dataset

DATA_ROOT = '/cluster/home/buec/eva/data'
OUT_PATH = '/cluster/home/buec/eva/output'

parser = argparse.ArgumentParser()

parser.add_argument('out_folder', type=str)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--no_cuda', action="store_true")
parser.add_argument('--small_balanced', action="store_true")

args = parser.parse_args()

#
#
#

out_folder = os.path.join(OUT_PATH, args.out_folder)
os.makedirs(out_folder, exist_ok=True)
utils.setup_logging(os.path.join(out_folder, 'log.txt'))

#
#
#

tr_path = os.path.join(DATA_ROOT, 'tr_split.txt')
dev_path = os.path.join(DATA_ROOT, 'dev_split.txt')

tr_list = textfile.read_separated_lines(tr_path, separator=' ')
dev_list = textfile.read_separated_lines(dev_path, separator=' ')

samples_path = os.path.join(DATA_ROOT, 'train_samples.h5')
feats = h5py.File(samples_path, 'r')

#
#
#

if args.small_balanced:
    random.shuffle(tr_list)
    random.shuffle(dev_list)

    tr_small = collections.defaultdict(list)
    dev_small = collections.defaultdict(list)

    for track in tr_list:
        if len(tr_small[str(track[2])]) < 200:
            tr_small[str(track[2])].append(track)

    for track in dev_list:
        if len(dev_small[str(track[2])]) < 50:
            dev_small[str(track[2])].append(track)

    tr_list = []
    for tracks in tr_small.values():
        tr_list.extend(tracks)

    dev_list = []
    for tracks in dev_small.values():
        dev_list.extend(tracks)

#
#
#

train_ds = dataset.RawDataset(tr_list, feats)
dev_ds = dataset.RawDataset(dev_list, feats)

train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
dev_loader = data.DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

#
#
#

model = model.ModelRaw(dropout=args.dropout)

model_checkpoint_path = os.path.join(out_folder, 'model_checkpoints')
model_checkpoint_cb = callbacks.ModelCheckpointCallback(model_checkpoint_path)

categorical_accuracy = metrics.CategoricalAccuracy('Categorical Accuracy')
mean_log_loss = utils.MeanLogLoss('Mean Log Loss')

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
bce = torch.nn.BCELoss()

trainer = candle.Trainer(model, optimizer,
                         targets=[candle.Target('BCE', bce)],
                         callbacks=[model_checkpoint_cb],
                         metrics=[categorical_accuracy, mean_log_loss],
                         num_epochs=args.num_epochs,
                         use_cuda=not args.no_cuda)

train_log = trainer.train(train_loader, dev_loader)

torch.save(model.state_dict(), os.path.join(out_folder, 'model.pytorch'))

#
#
#

logging.info('Store log data')
train_log.save_in_folder(os.path.join(out_folder, 'train_log'))
os.makedirs(os.path.join(out_folder, 'train_loss'), exist_ok=True)
os.makedirs(os.path.join(out_folder, 'train_metrics'), exist_ok=True)
train_log.save_loss_plots_at(os.path.join(out_folder, 'train_loss'))
train_log.save_metric_plots_at(os.path.join(out_folder, 'train_metrics'))

train_log.write_stats_to(os.path.join(out_folder, 'training_result.txt'))
