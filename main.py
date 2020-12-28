#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import logging
import argparse
# External imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import tqdm
import deepcs.display
from deepcs.training import train as ftrain, ModelCheckpoint
from deepcs.testing import test as ftest
from deepcs.fileutils import generate_unique_logpath
import deepcs.metrics
# Local imports
import data
import models


def wrap_args(packed_predictions, packed_targets):
    """
    Little wraper to drop the first element of the target
    """
    # The packed_targets virtualy "contain"
    #  <sos> c1 c2 c3 c4 .... <eos>
    # The predictions are expected to predict
    # c1 c2 c3 ... <eos>

    # Therefore, the targets to predict are in the slice
    #  targets[i, 1:li]
    # Which are to be compared with the probabilities in
    #  predictions[i, :li-1, ...]
    # predictions, lens_predictions = pad_packed_sequence(packed_predictions,
    #                                                     batch_first=True)
    targets, lens_targets = pad_packed_sequence(packed_targets,
                                                batch_first=True)
    # Remove the <sos> from the targets
    targets = targets[:, 1:]
    lens_targets -= 1
    # Repack it
    packed_targets = pack_padded_sequence(targets,
                                          lengths=lens_targets,
                                          enforce_sorted=False,
                                          batch_first=True)

    return packed_predictions.data, packed_targets.data


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Data loading
    train_loader, valid_loader, test_loader = data.get_dataloaders(args.datasetroot,
                                                                   args.datasetversion,
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
    baseloss = nn.CrossEntropyLoss()
    celoss = lambda *args: baseloss(* wrap_args(*args))
    accuracy = lambda *args: deepcs.metrics.accuracy(* wrap_args(*args))
    optimizer = optim.AdamW(model.parameters())

    metrics = {
        'CE': celoss,
        'accuracy': accuracy
    }

    # Callbacks
    summary_text = "Summary of the model architecture\n"+ \
            "=================================\n" + \
            f"{deepcs.display.torch_summarize(model)}\n"

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'seq2seq')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    model_checkpoint = ModelCheckpoint(model,
                                       os.path.join(logdir, 'best_model.pt'))

    # Training loop
    for e in range(args.num_epochs):
        ftrain(model,
               train_loader,
               celoss,
               optimizer,
               device,
               metrics,
               num_model_args=2,
               num_epoch=e,
               tensorboard_writer=tensorboard_writer)

        # Compute and record the metrics on the validation set
        valid_metrics = ftest(model,
                              valid_loader,
                              device,
                              metrics,
                              num_model_args=2)
        better_model = model_checkpoint.update(valid_metrics['CE'])
        logger.info("[%d/%d] Validation:   Loss : %.3f | Acc : %.3f%% %s"% (e,
                                                                         args.num_epochs,
                                                                         valid_metrics['CE'],
                                                                         100.*valid_metrics['accuracy'],
                                                                           "[>> BETTER <<]" if better_model else ""))

        for m_name, m_value in valid_metrics.items():
            tensorboard_writer.add_scalar(f'metrics/valid_{m_name}',
                                          m_value,
                                          e+1)
        # Compute and record the metrics on the test set
        test_metrics = ftest(model,
                             test_loader,
                             device,
                             metrics,
                             num_model_args=2)
        logger.info("[%d/%d] Test:   Loss : %.3f | Acc : %.3f%%"% (e,
                                                                   args.num_epochs,
                                                                   test_metrics['CE'],
                                                                   100.*test_metrics['accuracy']))
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f'metrics/test_{m_name}',
                                          m_value,
                                          e+1)

        # Try to decode some of the validation samples

        # And save them to the tensorboard



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
    parser.add_argument("--datasetversion",
                        choices=['v1', 'v6.1'],
                        default=data._DEFAULT_COMMONVOICE_VERSION,
                        help="Which CommonVoice corpus to consider")
    parser.add_argument("--datasetroot",
                        type=str,
                        default=data._DEFAULT_COMMONVOICE_ROOT,
                        help="The root directory holding the datasets. "
                        " These are supposed to be datasetroot/v1/fr or "
                        " datasetroot/v6.1/fr"
                       )
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
