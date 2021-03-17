"""
base_arg_parser.py
Base arguments for all scripts
"""

import argparse
import json
import os
from os.path import dirname, join
import random
import subprocess
from datetime import datetime

import torch
import numpy as np

class ArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='privacy')

        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible outputs.') #TODO seeds are set in training scripts; may be preferrable to set the seed in parse_args?

        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
        self.parser.add_argument('--viz_batch_size', type=int, default=4, help='Visualization image batch size.')

        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=1, type=int, help='Number of threads for the DataLoader.')

        self.parser.add_argument('--save_dir', type=str, default=join(os.curdir, dirname(__file__), 'results'), help='Directory for results, prefix.')
        self.parser.add_argument('--num_visuals', type=str, default=4, help='Number of visual examples to show per batch on Tensorboard.')

        self.parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
        self.parser.add_argument('--step_train_discriminator', type=float, default=1, help='Train discriminator every x steps. Set x here.')
        
        self.parser.add_argument('--max_ckpts', type=int, default=15, help='Max ckpts to save.')
        self.parser.add_argument('--load_path', type=str, default=None, help='Load from a previous checkpoint.')
        
        self.parser.add_argument('--steps_per_print', type=int, default=10, help='Steps taken for each print of logger')
        self.parser.add_argument('--steps_per_visual', type=int, default=100, help='Steps for  each visual to be printed by logger in tb')

    def parse_args(self):
        args = self.parser.parse_args()

        # Get version control hash for record-keeping
        args.commit_hash = subprocess.run(
            ['git', '-C', join('.', dirname(__file__)), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout.strip()
        # This appends, if necessary, a message about there being uncommitted changes
        # (i.e. if there are uncommitted changes, you can't be sure exactly
        # what the code looks like, whereas if there are no uncommitted changes,
        # you know exactly what the code looked like).
        args.commit_hash += ' (with uncommitted changes)' if bool(subprocess.run(
            ['git', '-C', join('.', dirname(__file__)), 'status', '--porcelain'], stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout.strip()) else ''

        # Create save dir for run
        args.name = (f'{args.name}_{datetime.now().strftime("%b%d_%H%M%S")}'
            f'_{os.getlogin()}')
        save_dir = os.path.join(args.save_dir, f'{args.name}')
        os.makedirs(save_dir, exist_ok=False)
        args.save_dir = save_dir

        # Create ckpt dir and viz dir
        args.ckpt_dir = os.path.join(args.save_dir, 'ckpts')
        os.makedirs(args.ckpt_dir, exist_ok=False)

        args.viz_dir = os.path.join(args.save_dir, 'viz')
        os.makedirs(args.viz_dir, exist_ok=False)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        if hasattr(args, 'supervised_factors'):
            args.supervised_factors = args_to_list(args.supervised_factors)

        # Save args to a JSON file
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        print(json.dumps(vars(args), indent=4, sort_keys=True))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        return args
