import logging

import numpy as np
import torch
import candle


def setup_logging(path):
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')

    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logging.getLogger().addHandler(fh)


class MeanLogLoss(candle.Metric):
    def compute(self, output, target, model=None):
        output = torch.clamp(output, min=1e-20)
        num = target.size(0)
        loss = -torch.sum(torch.mul(target, torch.log(output))) / num

        return loss, num

    def cumulate(self, metric_values=[]):
        data = np.stack(metric_values).T
        num = data[1].sum()
        loss = (data[0] * data[1]).sum() / num

        return loss, num

    @classmethod
    def plotable_columns(cls):
        return ['Mean Log Loss']

    @classmethod
    def columns(cls):
        return ['Mean Log Loss', 'Num Samples']
