#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import sys
import logging
import argparse
import functools
from pathlib import Path
# External imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchaudio
import tqdm
import deepcs.display
from deepcs.training import train as ftrain, ModelCheckpoint
from deepcs.testing import test as ftest
from deepcs.fileutils import generate_unique_logpath
import deepcs.metrics
# Local imports
import data
import models


def wrap_ctc_args(packed_predictions, packed_targets):
    """
    Returns:
        log_softmax predictions, targets, lens_predictions, lens_targets
    """
    unpacked_predictions, lens_predictions = pad_packed_sequence(packed_predictions,
                                                                 batch_first=True)
    # compute the log_softmax
    unpacked_predictions = unpacked_predictions.log_softmax(dim=2)
    # make it (T, batch, vocab_size)
    unpacked_predictions = unpacked_predictions.transpose(0, 1)

    unpacked_targets, lens_targets = pad_packed_sequence(packed_targets,
                                                        batch_first=True)

    return unpacked_predictions, unpacked_targets, lens_predictions, lens_targets


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Data loading
    loaders = data.get_dataloaders(args.datasetroot,
                                   args.datasetversion,
                                   cuda=use_cuda,
                                   batch_size=args.batch_size,
                                   n_threads=args.nthreads,
                                   small_experiment=args.debug,
                                   train_augment=args.train_augment,
                                   nmels=args.nmels,
                                   logger=logger)
    train_loader, valid_loader, test_loader = loaders

    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()

    # Model definition
    n_mels = args.nmels
    n_hidden_rnn = args.nhidden_rnn
    n_layers_rnn = args.nlayers_rnn
    blank_id = charmap.vocab_size

    num_model_args = 1
    model = models.CTCModel(charmap, n_mels, n_hidden_rnn, n_layers_rnn)
    decode = functools.partial(model.beam_decode, beam_size=10,
                               blank_id=blank_id)

    model.to(device)

    # Loss, optimizer
    baseloss = nn.CTCLoss(blank=blank_id)
    loss = lambda *args: baseloss(* wrap_ctc_args(*args))
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)

    metrics = {
        'CTC': loss
    }

    # Callbacks
    summary_text = "## Summary of the model architecture\n"+ \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'ctc')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    with open(os.path.join(logdir, "summary.txt"), 'w') as f:
        f.write(summary_text)

    model_checkpoint = ModelCheckpoint(model,
                                       os.path.join(logdir, 'best_model.pt'))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    for e in range(args.num_epochs):
        ftrain(model,
               train_loader,
               loss,
               optimizer,
               device,
               metrics,
               grad_clip=args.grad_clip,
               num_model_args=num_model_args,
               num_epoch=e,
               tensorboard_writer=tensorboard_writer)

        # Compute and record the metrics on the validation set
        valid_metrics = ftest(model,
                              valid_loader,
                              device,
                              metrics,
                              num_model_args=num_model_args)
        better_model = model_checkpoint.update(valid_metrics['CTC'])
        scheduler.step()

        logger.info("[%d/%d] Validation:   CTCLoss : %.3f %s"% (e,
                                                                args.num_epochs,
                                                                valid_metrics['CTC'],
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
                             num_model_args=num_model_args)
        logger.info("[%d/%d] Test:   Loss : %.3f "% (e,
                                                     args.num_epochs,
                                                     test_metrics['CTC']))
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f'metrics/test_{m_name}',
                                          m_value,
                                          e+1)
        # Try to decode some of the validation samples
        model.eval()
        decoding_results = "## Decoding results on the validation set\n"
        valid_batch = next(iter(valid_loader))
        spectro, transcripts = valid_batch
        spectro = spectro.to(device)
        # unpacked_spectro is (batch_size, seq_len, n_mels)
        unpacked_spectro, lens_spectro = pad_packed_sequence(spectro,
                                                             batch_first=True)
        # unpacked_transcripts is (batch_size, seq_len)
        unpacked_transcripts, lens_transcripts = pad_packed_sequence(transcripts,
                                                                     batch_first=True)
        # valid_batch is (batch, seq_len, n_mels)
        for idxv in range(5):
            spectrogram = unpacked_spectro[idxv, :, :].unsqueeze(dim=0)
            spectrogram = pack_padded_sequence(spectrogram, batch_first=True,
                                               lengths=[lens_spectro[idxv]])
            likely_sequences = decode(spectrogram)

            decoding_results += "\nGround truth : " + charmap.decode(unpacked_transcripts[idxv]) + '\n'
            decoding_results += "Log prob     Sequence\n"
            decoding_results += "\n".join(["{:.2f}        {}".format(p, s) for (p, s) in likely_sequences])
            decoding_results += '\n'
        print(decoding_results)
        tensorboard_writer.add_text("Decodings",
                                    deepcs.display.htmlize(decoding_results),
                                    global_step=e+1)


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
    parser.add_argument("--batch_size",
                       type=int,
                       help="The size of the minibatch",
                       default=64)
    parser.add_argument("--base_lr",
                        type=float,
                        help="The base learning rate for the optimizer",
                        default=0.001)
    parser.add_argument("--grad_clip",
                        type=float,
                        help="The maxnorm of the gradient to clip to",
                        default=None)
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to test on a small experiment")
    parser.add_argument("--num_epochs",
                        type=int,
                        help="The number of epochs to train for",
                        default=50)
    parser.add_argument("--train_augment",
                        action="store_true",
                        help="Whether to use or not SpecAugment during training")
    parser.add_argument("--nmels",
                        type=int,
                        help="The number of scales in the MelSpectrogram",
                        default=data._DEFAULT_NUM_MELS)
    parser.add_argument("--nlayers_rnn",
                        type=int,
                        help="The number of RNN layers",
                        default=3)
    parser.add_argument("--nhidden_rnn",
                        type=int,
                        help="The number of units per recurrent layer",
                        default=256)
    parser.add_argument("--weight_decay",
                        type=float,
                        help="The weight decay coefficient",
                        default=0.01)

    # For testing/decoding
    parser.add_argument("--modelpath",
                        type=Path,
                        help="The pt path to load")
    parser.add_argument("--audiofile",
                        type=Path,
                        help="The path to the audio file to transcript")

    args = parser.parse_args()

    eval(f"{args.command}(args)")
