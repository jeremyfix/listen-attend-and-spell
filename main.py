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
    n_mels = data._DEFAULT_NUM_MELS
    model = models.Model(n_mels)
    model.to(device)

    for X, y in tqdm.tqdm(train_loader):
        X, y = X.to(device), y.to(device)

        model(X)

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
