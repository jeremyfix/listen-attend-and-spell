#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import logging
import argparse
# External imports
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
import tqdm
import deepcs.display
from deepcs.training import train as train_epoch
from deepcs.testing import test
from deepcs.fileutils import generate_unique_logpath
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
                                                                   n_threads=args.nthreads,
                                                                   small_experiment=args.debug)
    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()
    vocab_size = charmap.vocab_size

    # Model definition
    n_mels = data._DEFAULT_NUM_MELS
    n_hidden_listen = args.nhidden_listen
    n_hidden_spell = args.nhidden_spell
    dim_embed = args.dim_embed
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

    # Callbacks
    summary_text = "Summary of the model architecture\n"+ \
                    "=================================\n" + \
                    f"{deepcs.display.torch_summarize(model)}\n"

    print(summary_text)
    logdir = generate_unique_logpath('./logs', 'seq2seq')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    for e in range(args.num_epochs):
        train_epoch(model,
                    train_loader,
                    celoss,
                    optimizer,
                    device,
                    metrics,
                    num_model_args=2,
                    num_epoch=e,
                    tensorboard_writer=tensorboard_writer)
        # Compute the metrics on the validation set


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
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to test on a small experiment")
    parser.add_argument("--num_epochs",
                        type=int,
                        help="The number of epochs to train for",
                        default=10)

    parser.add_argument("--nhidden_listen",
                        type=int,
                        help="The number of units per recurrent layer of "
                        "the encoder layer",
                        default=256)
    parser.add_argument("--nhidden_spell",
                        type=int,
                        help="The number of units per recurrent layer of the "
                        "spell module",
                        default=256)
    parser.add_argument("--dim_embed",
                        type=int,
                        help="The dimensionality of the embedding layer "
                        "for the input characters of the decoder",
                        default=128)
    args = parser.parse_args()

    eval(f"{args.command}(args)")
