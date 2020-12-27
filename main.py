#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import logging
import argparse
# External imports
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import tqdm
from deepcs.training import train as train_epoch
from deepcs.testing import test
# Local imports
import data
import models


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Data loading
    train_loader, valid_loader, test_loader = data.get_dataloaders(data._DEFAULT_COMMONVOICE_ROOT,
                                                                   cuda=use_cuda,
                                                                  n_threads=args.nthreads)
    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()
    vocab_size = charmap.vocab_size

    n_mels = data._DEFAULT_NUM_MELS
    n_hidden_listen = 256
    n_hidden_spell = 256
    dim_embed = 128
    model = models.Model(n_mels,
                         vocab_size,
                         n_hidden_listen,
                         dim_embed,
                         n_hidden_spell)
    model.to(device)

    # Loss, optimizer
    celoss = models.PackedCELoss()
    optimizer = optim.AdamW(model.parameters())

    metrics = {
        'CE': celoss
    }

    train_epoch(model, train_loader, celoss, optimizer, device, metrics,
               num_model_args=2)

def test(args):
    """
    Test function from a trained network and a wav sample
    """
    logger = logging.getLogger(__name__)
    logger.info("Test")


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'test'])
    parser.add_argument("--nthreads",
                       type=int,
                       help="The number of threads to use for loading the data",
                       default=4)

    args = parser.parse_args()

    eval(f"{args.command}(args)")
