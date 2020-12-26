#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import functools
from pathlib import Path
from typing import Union
# External imports
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram
# Download the tokenizer
import nltk
nltk.download('punkt')

_DEFAULT_COMMONVOICE_ROOT = "/opt/Datasets/CommonVoice/fr_79h_2019-02-25/fr"


def load_dataset(fold: str,
                 commonvoice_root: Union[str, Path]) -> torch.utils.data.Dataset:
    """
    Load the commonvoice dataset

    Args:
        fold (str): the fold to load, e.g. train, dev, test, validated, ..

    Returns:
        torch.utils.data.Dataset: ``dataset``
    """
    return COMMONVOICE(root=commonvoice_root,
                       tsv=fold+".tsv")


class BatchCollate(object):
    """
    Collator for the individual data to build up the minibatches
    """

    _DEFAULT_RATE = 48000  # Hz
    _DEFAULT_WIN_LENGTH = 25  # ms
    _DEFAULT_WIN_STEP = 15  # ms

    def __init__(self):
        nfft = int(self._DEFAULT_WIN_LENGTH * 1e-3 * self._DEFAULT_RATE)
        nstep = int(self._DEFAULT_WIN_STEP * 1e-3 * self._DEFAULT_RATE)
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=self._DEFAULT_RATE,
                           n_fft=nfft,
                           hop_length=nstep),
            AmplitudeToDB()
        )

    def __call__(self, batch):
        """
        Builds and return a minibatch of data as a tuple (inputs, targets)

        Returns:
            a tuple (spectros, targets) with :
                spectors : (Batch size, n_mels, time)
                targets : (Batch size, vocab_size, time)

        """
        # Extract the subcomponents
        waveforms = [w for w, _, _ in batch]
        rates = set([r for _, r, _ in batch])
        transcripts = [d['sentence'] for _, _, d in batch]
        # Retrieve the maximal length in the list of waveforms
        # max_len = max([w.shape[1] for w in waveforms])

        # Pad and stack the variable length tensors
        # waveforms is a 2D tensor (Batch, Time)
        waveforms = pad_sequence([t.squeeze() for t in waveforms],
                                 batch_first=True)

        spectrograms = self.transform(waveforms)

        if len(rates) != 1:
            raise NotImplementedError("Cannot deal with more than 1 sample rate in the data")
        if rates.pop() != self._DEFAULT_RATE:
            raise NotImplementedError("One batch is using a sampling rate different"
                                      f" from the assumed {self._DEFAULT_RATE} Hz")
        return spectrograms, transcripts



def get_dataloaders(commonvoice_root: str,
                    batch_size: int = 64,
                    n_threads: int = 4):
    """
    Build and return the pytorch dataloaders

    Args:
        commonvoice_root (str or Path) : the root directory where the dataset 
                                         is stored
        commonvoice_version (str or Path): the subdirectory
    """
    dataset_loader = functools.partial(load_dataset, commonvoice_root=commonvoice_root)
    train_dataset = dataset_loader("train")
    valid_dataset = dataset_loader("dev")
    test_dataset = dataset_loader("test")

    batch_collate_fn = BatchCollate()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=n_threads,
                                               collate_fn=batch_collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=n_threads,
                                               collate_fn=batch_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=n_threads,
                                               collate_fn=batch_collate_fn)

    return train_loader, valid_loader, test_loader
