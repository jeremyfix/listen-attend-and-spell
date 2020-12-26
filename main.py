#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import logging
import argparse
# External imports
import torch
import tqdm
# Local imports
import data
import models


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Data loading
    train_loader, valid_loader, test_loader = data.get_dataloaders(data._DEFAULT_COMMONVOICE_ROOT,
                                                                  n_threads=args.nthreads)
    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()
    vocab_size = charmap.vocab_size

    n_mels = data._DEFAULT_NUM_MELS
    n_hidden_listen = 38
    n_hidden_spell = 53
    model = models.Model(n_mels,
                         vocab_size,
                         n_hidden_listen,
                         n_hidden_spell)
    model.to(device)

    for X, y in tqdm.tqdm(train_loader):
        X, y = X.to(device), y.to(device)

        model(X, y)

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
