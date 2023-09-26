"""Main entry point for evaluation.

The model and model quantization params are specified in .yaml file (see 'config/example_eval.yaml').
Can be called from terminal to evaluate multiple backbones on multiple quantization techniques. For example: 
    python test.py --out path/to/out --cfg path/to/config.yaml --data path/to/dataset/root --targets calfw cfpfp
    --qfunc kmeans none --nc 64 32 --bs 512
"""


import os
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
import yaml

from utils.callbacks import CallBackVerification
from backbones.iresnet import iresnet50, iresnet18
from backbones.iresnet import quantize_model as quantize_resnet
from backbones.mobilefacenet import MobileFaceNet
from backbones.mobilefacenet import quantize_model as quantize_mobile
from quantization.modules import freeze_model
from quantization.colors import quantize_colors_division, quantize_colors_kmeans



def log_msg(msg):
    return f"[{datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}] {msg}"


def main(logfile, models, qparams, val_sets, data_root, qfunc, num_colors, device, batch_size=64):
    """Evaluates multiple pretrained backbones on set of datasets.
    
    Args:
        logfile: Text file where accuracies are written to.
        models: Dictionary mapping model identifier (String) to a tuple of (`model_arch`, cp_path).
            `model_arch` can be any of {iresnet50, iresnet18, mobilefacenet}.
        qparams: Dictionary mapping a model identifier (String) to a tuple (weight_bits, activation_bits).
        val_sets: List of dataset names (without file ending) used for evaluation.
        data_root: Directory containing all a `name`.bin file for each `name` in `val_sets`.
        qfunc: Function used for quantizing images. Should expect two inputs (image, num_colors) and return quantized image.
        num_colors: Number of colors after quantization.
    """

    callback_val = CallBackVerification(
        1, val_sets, data_root, None, image_size=(112, 112),
        qfunc=qfunc, num_colors_per_channel=num_colors, device=device,
    )

    with open(logfile, "w") as log:
        pass

    for name in models:
        print(log_msg(f"Validating {name}"))

        wq, aq = qparams[name]
        arch, path = models[name]

        if arch == "iresnet50":
            model = iresnet50(num_features=512, use_se=False)
        elif arch == "iresnet18":
            model = iresnet18(num_features=512, use_se=False)
        elif arch == "mobilefacenet":
            model = MobileFaceNet()

        if "mobilefacenet" in arch and wq is not None:
            model = quantize_mobile(model, weight_bit=wq, act_bit=aq)
        elif "iresnet" in arch and wq is not None:
            model = quantize_resnet(model, weight_bit=wq, act_bit=aq)

        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        freeze_model(model)

        with torch.no_grad():
            accs = callback_val.ver_test(model, -1, batch_size=batch_size)
        
        with open(logfile, "a") as log:
            for dataset, acc in accs.items():
                log.write(log_msg(f"{name} | {dataset}: {acc:5f}\n"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="cuda device")
    parser.add_argument("--out", type=str)
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--targets", type=str, nargs="+", default=None)
    parser.add_argument("--qfunc", type=str, nargs="+", help="either 'kmeans', 'uniform' or 'none'")
    parser.add_argument("--nc", type=int, nargs="+", help="Number of colors")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.cfg, "r") as fin:
        config = SimpleNamespace(**yaml.safe_load(fin))
    if not os.path.exists(os.path.join(args.out, "config.yaml")):
        with open(os.path.join(args.out, "config.yaml"), "w") as fout:
            for k, v in config.__dict__.items():
                fout.write(f"{k}: {v}\n")

    models = {}
    qparams = {}
    for name in config.__dict__:
        arch, path, wq, aq = getattr(config, name)

        if wq == "None":
            wq = None
        if aq == "None":
            aq = None
        
        models[name] = (arch, path)
        qparams[name] = (wq, aq)

    if args.targets is None:
        val_targets = [file[:-4] for file in os.listdir(args.data) if ".bin" == file[-4:]]
    else:
        val_targets = args.targets
    
    device = torch.device(args.device if (args.device is not None) else "cpu")

    for qfunc in args.qfunc:
        for n_colors in args.nc:
            if qfunc == "kmeans":
                qfunc_ = quantize_colors_kmeans
            elif qfunc == "uniform":
                qfunc_ = quantize_colors_division
            elif qfunc == "none":
                qfunc_ = lambda x, y: x

            logfile = os.path.join(args.out, f"{qfunc}_{n_colors}.log")

            main(logfile, models, qparams, val_targets, args.data, qfunc_, n_colors, device, batch_size=args.bs)
