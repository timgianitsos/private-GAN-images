import os
from pathlib import Path
from datetime import datetime
from time import time

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
from torchnet.meter import AverageValueMeter
import torch.nn.functional as F
import torchvision.utils as vutils


class BaseLogger(object):
    def __init__(self, args, dataset_len):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.num_visuals = args.num_visuals
        
        self.dataset_len = dataset_len

        # Tensorboard dir: /data2/braingan/results/tb/{name}/
        tb_dir = Path('/'.join(args.save_dir.split('/')[:-1])) / 'tb' / args.name
        os.makedirs(tb_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=tb_dir)

        # Path to log file
        self.log_filepath = os.path.join(self.save_dir, f'{args.name}.log')

        self.epoch = 0
        self.iter = 0

        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = 0
        self.iter_start_time = None
        self.epoch_start_time = None

    def _log_text(self, text_dict):
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)

    def _log_scalars(self, scalar_dict, print_to_stdout=True, unique_id=None):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write(f'[{k}: {v}]')
            k = k.replace('_', '/')  # Group in TensorBoard by split.
            if unique_id is not None:
                k = f'{k}/{unique_id}'
            self.summary_writer.add_scalar(k, v, self.global_step)

    def visualize_gan_from_latent_vector(self, real, fake, split, unique_id=None):
        batch_size, _, _, _ = real.shape

        for i in range(self.num_visuals):
            if i >= batch_size:
                # Exceeded number in batch
                break

            # Get the i-th volume in batch
            real_vol = torch.squeeze(real[i], 0)
            fake_vol = torch.squeeze(fake[i], 0)

            vis_list = []
            # One slice at a time
            stacked_slice = torch.stack([real_vol.unsqueeze(0),
                                         fake_vol.unsqueeze(0),])

            # Concat along the width (dim=1)
            vis = torch.cat([ss.permute(1,2,0) for ss in stacked_slice], 1)
            vis_list.append(vis)

            # Concat along the height (dim=0)
            visuals = torch.cat([v.squeeze(-1) for v in vis_list])
            visuals_np = visuals.detach().to('cpu').numpy() * 255

            title = 'real-fake'
            tag = f'{split}/{title}'
            if unique_id is not None:
                tag += f'_{unique_id}'

            self.summary_writer.add_image(tag,
                                          np.uint8(visuals_np),
                                          self.global_step,
                                          dataformats='HW')

        return

    def visualize(self, inputs, targets, generated, split, unique_id=None):
        """Visualize predictions and targets in TensorBoard.

        Args:
            inputs: Inputs to the model.
            targets: Target images for the inputs.
            generated: Generated images from the inputs.
            split: One of 'train' or 'test'.
            unique_id: A unique ID to append to every image title. Allows
              for displaying all visualizations separately on TensorBoard.

        Returns:
            Number of examples visualized to TensorBoard.
        """
        batch_size, _, num_slices, _, _ = inputs.shape

        for i in range(self.num_visuals):
            if i >= batch_size:
                # Exceeded number in batch
                break
            
            # Get the i-th volume in batch
            inputs_vol = torch.squeeze(inputs[i], 0)
            targets_vol = torch.squeeze(targets[i], 0)
            generated_vol = torch.squeeze(generated[i], 0)
            
            vis_list = []
            for s in range(num_slices):
                # One slice at a time
                stacked_slice = torch.stack([inputs_vol[s].unsqueeze(0),
                                             targets_vol[s].unsqueeze(0),
                                             generated_vol[s].unsqueeze(0)])
                
                # Concat along the width (dim=1)
                vis = torch.cat([ss.permute(1,2,0) for ss in stacked_slice], 1)
                vis_list.append(vis)

            # Concat along the height (dim=0)
            visuals = torch.cat([v.squeeze(-1) for v in vis_list])
            visuals_np = visuals.detach().to('cpu').numpy() * 255

            title = 'inputs-targets-generated'
            tag = f'{split}/{title}'
            if unique_id is not None:
                tag += f'_{unique_id}'

            self.summary_writer.add_image(tag,
                                          np.uint8(visuals_np),
                                          self.global_step,
                                          dataformats='HW')

        return

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_filepath, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""

    def __init__(self, args, dataset_len, phase=None, using_gan=True):
        super(TrainLogger, self).__init__(args, dataset_len)
       
        # Tag suffix used for indicating training phase in loss + viz
        self.tag_suffix = phase
        
        self.num_epochs = args.num_epochs
        self.split = 'train'

        if using_gan:
            self.gen_loss_meter = AverageValueMeter()
            self.disc_loss_meter = AverageValueMeter()
        else:
            self.classifier_loss_meter = AverageValueMeter()
        
        self.device = args.device
        self.iter = 0
        self.steps_per_print = args.steps_per_print
        self.steps_per_visual = args.steps_per_visual

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter_classifier(self, batch_size, cls_loss):
        loss = cls_loss.item()
        self.classifier_loss_meter.add(loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / batch_size
            message = f"[epoch: {self.epoch}, iter: {self.iter} / {self.dataset_len}, time: {avg_time:.2f}, loss: {self.classifier_loss_meter.mean:.3g}]"
            self.write(message)

            # Write all errors as scalars to the graph
            self._log_scalars({'Loss': self.classifier_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.classifier_loss_meter.reset()

    def log_iter_gan_from_latent_vector(self, real, fake, gen_loss, disc_loss, privacy_eps=-1):
        """Log results from a training iteration."""
        batch_size = real.size(0)

        gen_loss = gen_loss.item()
        self.gen_loss_meter.add(gen_loss)

        if disc_loss is not None:
            disc_loss = disc_loss.item()
            self.disc_loss_meter.add(disc_loss)
        else:
            self.disc_loss_meter.add(-1)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / batch_size
            message = f"[epoch: {self.epoch}, iter: {self.iter} / {self.dataset_len}, time: {avg_time:.2f}, gen loss: {self.gen_loss_meter.mean:.3g}, disc loss: {self.disc_loss_meter.mean:.3g}, eps: {privacy_eps: .3g}]"
            self.write(message)

            # Write all errors as scalars to the graph
            self._log_scalars({'Loss_Gen': self.gen_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.gen_loss_meter.reset()

            self._log_scalars({'Loss_Disc': self.disc_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.disc_loss_meter.reset()

            if privacy_eps > 0:
                self._log_scalars({'eps': privacy_eps},
                                  print_to_stdout=False,
                                  unique_id=self.tag_suffix)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.steps_per_visual == 0:
            self.visualize_gan_from_latent_vector(real, fake, self.split, unique_id=self.tag_suffix)

    def log_iter(self, inputs, targets, generated, gen_loss, disc_loss):
        """Log results from a training iteration."""
        batch_size = inputs.size(0)

        gen_loss = gen_loss.item()
        self.gen_loss_meter.add(gen_loss, batch_size)

        if disc_loss is not None:
            disc_loss = disc_loss.item()
            self.disc_loss_meter.add(disc_loss, batch_size)
        else:
            self.disc_loss_meter.add(-1, batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / batch_size
            message = f"[epoch: {self.epoch}, iter: {self.iter} / {self.dataset_len}, time: {avg_time:.2f}, gen loss: {self.gen_loss_meter.mean:.3g}, disc loss: {self.disc_loss_meter.mean:.3g}]"
            self.write(message)

            # Write all errors as scalars to the graph
            self._log_scalars({'Loss_Gen': self.gen_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.gen_loss_meter.reset()

            self._log_scalars({'Loss_Disc': self.disc_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.disc_loss_meter.reset()

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.steps_per_visual == 0:
            self.visualize(inputs, targets, generated, self.split, unique_id=self.tag_suffix)

    def log_metrics(self, metrics):
        self._log_scalars(metrics)

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += 1
        self.global_step += 1

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write(f'[start of epoch {self.epoch}]')

    def end_epoch(self):
        """Log info for end of an epoch."""
        epoch_time = time() - self.epoch_start_time
        self.write(f'[end of epoch {self.epoch}, epoch time: {epoch_time:.2g}]')
        self.epoch += 1

