#!/usr/bin/env python3

# Standard imports
import logging
import argparse
# External imports

def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger()
    logger.info("Training")

def test(args):
    """
    Test function from a trained network and a wav sample
    """
    logger = logging.getLogger()
    logger.info("Test")


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'test'])

    args = parser.parse_args()

    eval(f"{args.command}(args)")
