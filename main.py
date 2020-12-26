#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import logging
import argparse
# External imports
# Local imports
import data


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    # Data loading
    train_loader, valid_loader, test_loader = data.get_dataloaders(data._DEFAULT_COMMONVOICE_ROOT,
                                                                  n_threads=args.nthreads)

    X, y = next(iter(train_loader))
    print(X.shape)



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
