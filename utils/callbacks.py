import logging
import os
import time

import torch

from eval import verification
from quantization.modules import unfreeze_model, freeze_model



class CallBackVerification:

    def __init__(self, frequency, val_targets, root_dir, output, image_size=(112, 112), qfunc=None, num_colors_per_channel=256, device=None):
        self.frequency = frequency
        self.output = output
        self.device = device
        self.datasets = {k: None for k in val_targets}
        self.highest_accs = {k: None for k in val_targets}
        if qfunc is not None:
            color_transform = lambda img: qfunc(img, num_colors_per_channel)
        else:
            color_transform = None
        self.init_dataset(root_dir, image_size, color_transform)

    def init_dataset(self, data_dir, image_size, qfunc):
        for name in self.datasets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                print("=> Loading bin", path)
                data_set = verification.load_bin(path, image_size, qfunc)
                self.datasets[name] = data_set

    def ver_test(self, backbone, global_step, batch_size=64):
        results = {}
        for name in self.datasets:
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(self.datasets[name], backbone, batch_size, 10, self.device)

            logging.info(f"[{global_step}] '{name}': xnorm={xnorm} | accuracy-flip={acc2:1.5f}+-{std2:1.5f}")

            if self.highest_accs[name] is None or acc2 > self.highest_accs[name]:
                self.highest_accs[name] = acc2

                if self.output is not None:
                    path = os.path.join(self.output, f"best_{name}.pt")
                    torch.save(backbone.module.state_dict(), path)

            logging.info(f"[{global_step}] '{name}': highest-acc={self.highest_accs[name]:1.5f}")
            results[name] = acc2

        return results

    @torch.no_grad()
    def __call__(self, epoch, global_step, backbone, force=False, unfreeze=True):
        timing = (epoch+1)%self.frequency == 0
        if timing or force:
            backbone.eval()
            freeze_model(backbone)

            self.ver_test(backbone, global_step)
            
            backbone.train()
            if unfreeze:
                unfreeze_model(backbone)


class CallBackLogging:

    def __init__(self, frequency,
        batch_size, world_size, num_epochs, epoch_len, writer=None
    ):
        self.frequency = frequency
        self.time_start = time.time()
        self.batch_size = batch_size
        self.world_size = world_size
        self.writer = writer
        self.tic = None
        self.total_epoch = num_epochs
        self.epoch_len = epoch_len

    def __call__(self, batch, epoch, global_step, loss, remaining_batches):
        if (global_step+1)%self.frequency == 0:
            if self.tic is not None:
                speed = self.frequency*self.batch_size / (time.time() - self.tic)
                speed_total = speed * self.world_size
            else:
                speed = float('inf')
                speed_total = float('inf')

            time_for_end = remaining_batches * self.batch_size / speed / 3600

            if self.writer is not None:
                self.writer.add_scalar('time_for_end', time_for_end, global_step)
                self.writer.add_scalar('loss', loss.avg, global_step)

            msg = f"[{global_step}]  Loss: {loss.avg:.4f}   Epoch: {epoch+1}/{self.total_epoch}   " \
                f"Batch: {batch+1}/{self.epoch_len}   Speed: {speed:.2f} (total_speed: {speed_total:.2f})   " \
                f"Time_remaining: {time_for_end:.2f}"

            logging.info(msg)
            loss.reset()
            self.tic = time.time()


class CallBackCheckpoint:

    def __init__(self, frequency, output):
        self.output = output
        self.frequency = frequency

    def __call__(self, epoch, backbone, header, rng, scheduler_backbone,
                 scheduler_header, optim_backbone, optim_header, quantized):
        if (epoch+1)%self.frequency == 0:
            cp = {
                "epoch": epoch,
                "backbone": backbone.module.state_dict(),
                "header": header.module.state_dict(),
                "optim_backbone": optim_backbone.state_dict(),
                "optim_header": optim_header.state_dict(),
                "sampling_generator": rng.get_state(),
                "quantized": quantized,
            }

            if scheduler_backbone is not None:
                cp["scheduler_backbone"] = scheduler_backbone.state_dict()
            if scheduler_header is not None:
                cp["scheduler_header"] = scheduler_header.state_dict()

            path = os.path.join(self.output, f"epoch_{epoch}.tar")
            torch.save(cp, path)
