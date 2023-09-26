import logging
import os
import sys
from types import SimpleNamespace

import yaml

from quantization.colors import quantize_colors_division, quantize_colors_kmeans


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(log_file, logger):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %I:%M:%S %p')
    handler_file = logging.FileHandler(log_file, mode="a")
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)


def dump_config(file, config):
    with open(file, "w") as fout:
        for k, v in config.__dict__.items():
            if callable(v):
                if v.__name__ == quantize_colors_kmeans.__name__:
                    v = "kmeans"
                elif v.__name__ == quantize_colors_division.__name__:
                    v = "uniform"
                elif v.__name__ == "lr_step":
                    v = "step"
            elif v is None:
                v = "none"

            fout.write(f"{k}: {v}\n")


def load_config(file):
    with open(file, "r") as fin:
        config = yaml.safe_load(fin)

    qfunc = config["qfunc"]
    if qfunc == "kmeans":
        qfunc = quantize_colors_kmeans
    elif qfunc == "uniform":
        qfunc = quantize_colors_division
    config["qfunc"] = qfunc

    lr_func = config["lr_func"]
    if lr_func == "step":
        def lr_step(epoch):
            return ((epoch+1)/(5)**2 if epoch < -1 else 0.1**len([
                m for m in [3, 5, 7] if (m-1 <= epoch)
            ]))
        lr_func = lr_step
    config["lr_func"] = lr_func

    for k in config:
        if config[k] == "none":
            config[k] = None
        
    return SimpleNamespace(**config)
