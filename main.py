#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import sys
import logging
import argparse
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

import matplotlib.pyplot as plt
# Local imports
import data
import models


def wrap_args(packed_predictions, packed_targets):
    """
    Little wraper to drop the first element of the target
    And to slice and return the packed_predictions and packed_targets
    as (batch_size*seq_len, vocab_size) and (batch*seq_len, ) tensors

    This functions takes care of slicing the sequences given their respective
    lengths

    Args:
        packed_predictions (PackedSequence) : (batch, seq_len, vocab_size)
        packed_targets (PackedSequence) : (batch, seq_len)
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
    unpacked_targets, lens_targets = pad_packed_sequence(packed_targets,
                                                         batch_first=True)
    # Remove the <sos> from the targets
    unpacked_targets = unpacked_targets[:, 1:]
    lens_targets -= 1

    targets = torch.hstack([
        targeti[:leni] for targeti, leni in zip(unpacked_targets,
                                                lens_targets)
    ])

    unpacked_predictions, lens_predictions = pad_packed_sequence(packed_predictions,
                                                                 batch_first=True)

    predictions = torch.vstack([
        predi[:leni] for predi, leni in zip(unpacked_predictions,
                                            lens_targets)
    ])

    return predictions, targets


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
                                                                   batch_size=args.batch_size,
                                                                   n_threads=args.nthreads,
                                                                   small_experiment=args.debug,
                                                                   nmels=args.nmels)
    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()

    # Model definition
    n_mels = args.nmels
    n_hidden_listen = args.nhidden_listen
    n_layers_encoder = args.nlayers_encoder
    n_hidden_spell = args.nhidden_spell
    n_layers_decoder = args.nlayers_decoder
    dim_embed = args.dim_embed
    model = models.Seq2Seq(n_mels,
                           charmap,
                           n_hidden_listen,
                           n_layers_encoder,
                           dim_embed,
                           n_hidden_spell,
                           n_layers_decoder,
                           args.teacher_forcing)
    model.to(device)

    # Loss, optimizer
    baseloss = nn.CrossEntropyLoss()
    celoss = lambda *args: baseloss(* wrap_args(*args))
    accuracy = lambda *args: deepcs.metrics.accuracy(* wrap_args(*args))
    optimizer = optim.Adam(model.parameters())

    metrics = {
        'CE': celoss,
        'accuracy': accuracy
    }

    # Callbacks
    summary_text = "## Summary of the model architecture\n"+ \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'seq2seq')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    with open(os.path.join(logdir, "summary.txt"), 'w') as f:
        f.write(summary_text)

    model_checkpoint = ModelCheckpoint(model,
                                       os.path.join(logdir, 'best_model.pt'))
    scheduler = lr_scheduler(optimizer, step_size=1, gamma=0.5)

    # Training loop
    for e in range(args.num_epochs):
        if e == 5 and args.teacher_forcing:
            logger.info("Disabling teacher forcing")
            model.set_forcing(False)

        ftrain(model,
               train_loader,
               celoss,
               optimizer,
               device,
               metrics,
               grad_clip=args.grad_clip,
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
        scheduler.step()

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
        model.eval()
        decoding_results = "## Decoding results on the validation set\n"
        valid_batch = next(iter(valid_loader))
        spectro, transcripts = valid_batch
        spectro = spectro.to(device)
        # unpacked_spectro is (batch_size, seq_len, n_mels)
        unpacked_spectro, lens_spectro = pad_packed_sequence(spectro, batch_first=True)
        # unpacked_transcripts is (batch_size, seq_len)
        unpacked_transcripts, lens_transcripts = pad_packed_sequence(transcripts, batch_first=True)
        # valid_batch is (batch, seq_len, n_mels)
        for idxv in range(5):
            spectrogram = unpacked_spectro[idxv, :, :].unsqueeze(dim=0)
            spectrogram = pack_padded_sequence(spectrogram, batch_first=True,
                                               lengths=[lens_spectro[idxv]])
            transcript = unpacked_transcripts[idxv, :].unsqueeze(dim=0)
            transcript = pack_padded_sequence(transcript, batch_first=True,
                                              lengths=[lens_transcripts[idxv]])
            likely_sequences = model.decode(args.beamwidth,
                                            args.maxlength,
                                            spectrogram)

            decoding_results += "\nGround truth : " + charmap.decode(unpacked_transcripts[idxv]) + '\n'
            decoding_results += "Log prob     Sequence\n"
            decoding_results += "\n".join(["{:.2f}        {}".format(p, s) for (p, s) in likely_sequences])
            decoding_results += '\n'
        print(decoding_results)
        tensorboard_writer.add_text("Decodings", deepcs.display.htmlize(decoding_results))




def test(args):
    """
    Test function from a trained network and a wav sample
    """
    logger = logging.getLogger(__name__)
    logger.info("Test")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()

    # Create the model
    # It is required to build up the same architecture than the one
    # used during training. If you do not remember the parameters
    # check the summary.txt file in the logdir where you have you
    # modelpath pt file saved
    n_mels = args.nmels
    n_hidden_listen = args.nhidden_listen
    n_hidden_spell = args.nhidden_spell
    dim_embed = args.dim_embed

    logger.info("Building the model")
    model = models.Seq2Seq(n_mels,
                           charmap,
                           n_hidden_listen,
                           dim_embed,
                           n_hidden_spell,
                           teacher_forcing=None)
    model.to(device)
    model.load_state_dict(torch.load(args.modelpath))

    # Switch the model to eval mode
    model.eval()

    # Load and preprocess the audiofile
    logger.info("Loading and preprocessing the audio file")
    waveform, sample_rate = torchaudio.load(args.audiofile)
    waveform_processor = data.WaveformProcessor(n_mels)
    spectrogram = waveform_processor(waveform).to(device)
    spectro_length = spectrogram.shape[1]

    # Plot the spectrogram
    logger.info("Plotting the spectrogram")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(spectrogram[0].cpu().numpy(),
              aspect='equal', cmap='magma', origin='lower')
    ax.set_xlabel("Mel scale")
    ax.set_ylabel("Time (sample)")
    fig.tight_layout()
    plt.savefig("spectro_test.png")

    spectrogram = pack_padded_sequence(spectrogram,
                                       lengths= [spectro_length],
                                       batch_first=True)

    logger.info("Decoding the spectrogram")
    likely_sequences = model.decode(args.beamwidth, args.maxlength, spectrogram)
    print("Log prob    Sequence\n")
    print("\n".join(["{:.2f}      {}".format(p, s) for (p, s) in likely_sequences]))


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
                        default=10)
    parser.add_argument("--nmels",
                        type=int,
                        help="The number of scales in the MelSpectrogram",
                        default=data._DEFAULT_NUM_MELS)
    parser.add_argument("--nlayers_encoder",
                        type=int,
                        help="The number of RNN layers for the encoder",
                        default=1)
    parser.add_argument("--nhidden_listen",
                        type=int,
                        help="The number of units per recurrent layer of "
                        "the encoder layer",
                        default=256)
    parser.add_argument("--nlayers_decoder",
                        type=int,
                        help="The number of RNN layers for the decoder",
                        default=1)
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
    parser.add_argument("--weight_decay",
                        type=float,
                        help="The weight decay coefficient",
                        default=0.01)
    parser.add_argument("--teacher_forcing",
                        action="store_true",
                        help="Whether or not to use teacher forcing")

    # For testing/decoding
    parser.add_argument("--modelpath",
                        type=Path,
                        help="The pt path to load")
    parser.add_argument("--audiofile",
                        type=Path,
                        help="The path to the audio file to transcript")
    parser.add_argument("--beamwidth",
                        type=int,
                        default=3,
                        help="The number of alternatives to consider when"
                        " for beam search decoding")
    parser.add_argument("--maxlength",
                        type=int,
                        default=100,
                        help="The maximum length of the decoded string if no"
                        " <eos> is predicted.")


    args = parser.parse_args()

    eval(f"{args.command}(args)")
