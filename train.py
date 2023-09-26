"""Main entry point for training resnet/mobilefacenet on color quantized dataset.

Supports (model) quantization aware training and multi GPU training.

Should be called torchrun. Hyperparameter aswell as output folder
are specified in a .yaml file (see 'config/example_training.yaml'). If any checkpoint exists in output folder the training
is resumed from the latest checkpoint.

Training for a single configuration:
    torchrun --standalone --nnodes=1 --nproc-per-node=<number-of-gpus> ./train.py path/to/config.yaml

Training for multiple configs, collected in a single directory:
    bash ./run.sh path/to/config_dir/
"""


import os
import random
import logging
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel

from backbones.iresnet import iresnet100, iresnet18, iresnet50
from backbones.mobilefacenet import MobileFaceNet
from quantization.modules import unfreeze_model, freeze_model
from utils import losses
from utils.dataset import FaceDatasetFolder, MXFaceDataset
from utils.utils_logging import load_config, dump_config, AverageMeter, init_logging
from utils.callbacks import CallBackLogging, CallBackCheckpoint, CallBackVerification



def seed_all(seed):
    """Enables deterministic training.

    Seeds all random generators used and disables all pytorch implementations that are
    non-deterministic.

    Args:
        seed: Integer used for random generators.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main(config, checkpoint=None):
    """Trains single backbone on multiple GPUS.

    Args:
        config: A dictionary (as returned by `utils.utils_logging.load_config`) containing configurations for training.
        checkpoint: Path to checkpoint to resume from (as String).
    """

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    
    if checkpoint is not None:
        resume = True

        cp = torch.load(checkpoint, map_location=f"cuda:{local_rank}")
        curr_epoch = cp["epoch"]+1  # checkpoint is taken at end of epoch
        quantized = cp["quantized"]
    else:
        cp = {}
        resume = False
        quantized = False
        curr_epoch = 0

    if rank == 0:
        seed_all(config.seed)

        print("Save Location:", os.path.abspath(config.output))
        
        # Logging
        logger = logging.getLogger()
        init_logging(os.path.join(config.output, f"{config.network}.log"), logger)
        
        if resume:
            logging.info(f"Resuming from: {checkpoint}")
        else:
            dump_config(os.path.join(config.output, "config.yaml"), config)
    
        logging.info("==== Config ====")
        d = config.__dict__
        for k in d.keys():
            logging.info(f"\t {k} = {d[k]}")
        logging.info("==== \t ====")


    if config.trainset == "mxnet":
        trainset = MXFaceDataset(config.train_root, config.qfunc, config.num_colors)
    else:
        trainset = FaceDatasetFolder(root_dir=config.train_root)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=trainset, rank=rank, num_replicas=world_size, shuffle=config.shuffle,
    )

    rng_sampling = torch.Generator()
    if resume and ("sampling_generator" in cp):
        rng_sampling.set_state(cp["sampling_generator"].cpu())
    else:
        rng_sampling.manual_seed(config.seed)

    def worker_init(id):
        base_seed = torch.initial_seed()
        seed = min(base_seed+id, 2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True, generator=rng_sampling,
        worker_init_fn=worker_init, prefetch_factor=2,
    )

    # load model
    if config.network == "iresnet100":
        backbone_ = iresnet100(num_features=config.embedding_size, use_se=config.SE).to(local_rank)
    elif config.network == "iresnet50":
        backbone_ = iresnet50(dropout=0.4, num_features=config.embedding_size, use_se=config.SE).to(local_rank)
    elif config.network == "iresnet18":
        backbone_ = iresnet18(dropout=0.4, num_features=config.embedding_size, use_se=config.SE).to(local_rank)
    elif config.network == "mobilefacenet":
        backbone_ = MobileFaceNet(embedding_size=config.embedding_size).to(local_rank)

    if "mobilefacenet" in config.network:
        from backbones.mobilefacenet import quantize_model
    else:
        from backbones.iresnet import quantize_model

    backbone = copy.deepcopy(backbone_)
    if resume and ("backbone" in cp):
        if quantized:
            backbone = quantize_model(backbone_, weight_bit=config.wq, act_bit=config.aq).to(local_rank)
            if rank == 0:
                logging.info("Quantize model")
        backbone.load_state_dict(cp["backbone"])
        unfreeze_model(backbone)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank]
    )
    backbone.train()


    # load header
    if config.header == "ElasticArcFace":
        header = losses.ElasticArcFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m, std=config.std).to(local_rank)
    elif config.header == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m, std=config.std, plus=True).to(local_rank)
    elif config.header == "ElasticCosFace":
        header = losses.ElasticCosFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m, std=config.std).to(local_rank)
    elif config.header == "ElasticCosFacePlus":
        header = losses.ElasticCosFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m,
                                       std=config.std, plus=True).to(local_rank)
    elif config.header == "ArcFace":
        header = losses.ArcFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m).to(local_rank)
    elif config.header == "CosFace":
        header = losses.CosFace(in_features=config.embedding_size, out_features=config.num_classes, s=config.s, m=config.m).to(
            local_rank)

    if resume and ("header" in cp):
        header.load_state_dict(cp["header"])

    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.eval()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=config.lr / 512 * config.batch_size * world_size,
        momentum=config.momentum, weight_decay=config.weight_decay)

    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=config.lr / 512 * config.batch_size * world_size,
        momentum=config.momentum, weight_decay=config.weight_decay)
        
    if resume and ("optim_backbone" in cp):
        opt_backbone.load_state_dict(cp["optim_backbone"])
    if resume and ("optim_header" in cp):
        opt_header.load_state_dict(cp["optim_header"])


    if config.lr_func is not None:
        scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt_backbone, lr_lambda=config.lr_func)
        
        scheduler_header = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt_header, lr_lambda=config.lr_func)  

        if resume and ("scheduler_backbone" in cp):
            scheduler_backbone.load_state_dict(cp["scheduler_backbone"])

        if resume and ("scheduler_header" in cp):
            scheduler_header.load_state_dict(cp["scheduler_header"])
    else:
        scheduler_backbone = None
        scheduler_header = None

    del cp


    if config.criterion == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif config.criterion == "mseloss":
        criterion = nn.MSELoss()

    total_step = len(train_loader) * config.epochs  # len gives batches per gpu
    if rank == 0:
        if not resume:
            logging.info(f"Total Step is: {total_step}")

        if header is None:
            logging.info(f"[{global_step}] Epoch={curr_epoch}: Knowledge distillation enabled!")

    if rank == 0:
        callback_logging = CallBackLogging(config.log_freq, config.batch_size, world_size, config.epochs, len(train_loader), writer=None)
        callback_checkpoint = CallBackCheckpoint(config.cp_freq, os.path.join(config.output, "checkpoints"))
        callback_val = CallBackVerification(config.val_freq, config.val_targets, config.val_root, config.output, qfunc=config.qfunc, device=local_rank, num_colors_per_channel=config.num_colors)
    
    loss = AverageMeter()
    global_step = curr_epoch * len(train_loader)
    frozen = False
    epoch = curr_epoch
    for epoch in range(curr_epoch, config.epochs):
        train_sampler.set_epoch(epoch)
        loss_norm = min(len(train_loader), config.bw_step)
        if config.qepoch is not None and (not quantized) and epoch >= config.qepoch:
            backbone = quantize_model(backbone, weight_bit=config.wq, act_bit=config.aq).to(local_rank)
            backbone = unfreeze_model(backbone)
            quantized = True
            if rank == 0:
                logging.info("Quantize model")

        if config.finetune_epoch is not None and (not frozen) and epoch >= config.finetune_epoch:
            freeze_model(backbone)
            frozen = True
            if rank == 0:
                logging.info("Freeze qparams model")

        opt_backbone.zero_grad()
        opt_header.zero_grad()

        for batch, (img, label) in enumerate(train_loader):
            if rank == 0 and batch == len(train_loader)//2:
                callback_val(epoch, global_step, backbone, unfreeze=(not frozen))

            img = img.cuda(local_rank, non_blocking=True)
            features = F.normalize(backbone(img))

            label = label.cuda(local_rank, non_blocking=True)
            features = header(features, label)

            loss_v = criterion(features, label) / loss_norm
            loss.update(loss_v.item(), 1)
            
            if (batch+1)%config.bw_step == 0 or batch == len(train_loader)-1:
                loss_v.backward()
                
                nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

                opt_backbone.step()
                opt_header.step()

                opt_backbone.zero_grad()
                opt_header.zero_grad()

                loss_norm = min(len(train_loader)-batch-1, config.bw_step)
            else:
                with backbone.no_sync():
                    loss_v.backward()

            if rank == 0:
                callback_logging(batch, epoch, global_step, loss, len(train_loader)-batch)
            
            global_step += 1

        if scheduler_backbone is not None:
            scheduler_backbone.step()
        if scheduler_header is not None:
            scheduler_header.step()

        if rank == 0:
            callback_checkpoint(
                epoch=epoch,
                backbone=backbone,
                header=header,
                rng=rng_sampling,
                scheduler_backbone=scheduler_backbone,
                scheduler_header=scheduler_header,
                optim_backbone=opt_backbone,
                optim_header=opt_header,
                quantized=quantized
            )

            callback_val(epoch, global_step, backbone, unfreeze=(not frozen))


    if rank == 0:
        callback_val(epoch, global_step, backbone, force=True)
        callback_checkpoint(
                    epoch=epoch,
                    backbone=backbone,
                    header=header,
                    rng=rng_sampling,
                    scheduler_backbone=scheduler_backbone,
                    scheduler_header=scheduler_header,
                    optim_backbone=opt_backbone,
                    optim_header=opt_header,
                    quantized=quantized
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, nargs=1, default=None)

    args = parser.parse_args()

    config = load_config(args.cfg[0])
    checkpoint = None
    cp_path = os.path.join(config.output, "checkpoints")
    if os.path.exists(config.output):
        try:
            checkpoint = [
                file for file in os.listdir(os.path.join(cp_path))
            ]
        except FileNotFoundError:
            checkpoint = None

        if len(checkpoint) >= 1:
            checkpoint = sorted(checkpoint, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
            checkpoint = os.path.join(cp_path, checkpoint)
        else:
            checkpoint = None
    else:
        os.makedirs(os.path.join(config.output, "checkpoints"), exist_ok=False)

    main(config=config, checkpoint=checkpoint)
    