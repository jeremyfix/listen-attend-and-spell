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


class CharMap(object):

    _SOS = 182
    _EOS = 166
    _PAD = 95

    def __init__(self):
        ord_chars = frozenset().union(
            range(97, 123),  # a-z
            range(48, 58),   # 0-9
            [32, 39, 44, 46],  # <space> <,> <.> <'>
            [self._SOS],  # <sos>¶
            [self._EOS],  # <eos>¦
            [10060],  # <unk> ❌
        )

        # The pad symbol is added first to guarantee it has idx 0
        self.idx2char = [chr(self._PAD)] + [chr(i) for i in ord_chars]
        self.char2idx = {
            c: idx for (idx, c) in enumerate(self.idx2char)
        }

        self.equivalent_char = {}
        for i in range(224, 229):
            self.equivalent_char[chr(i)] = 'a'
        for i in range(232, 236):
            self.equivalent_char[chr(i)] = 'e'
        for i in range(236, 240):
            self.equivalent_char[chr(i)] = 'i'
        for i in range(242, 247):
            self.equivalent_char[chr(i)] = 'o'
        for i in range(249, 253):
            self.equivalent_char[chr(i)] = 'u'
        # Remove the punctuation marks
        for c in ['!', '?', ';']:
            self.equivalent_char[c] = '.'
        for c in ['-', '…', ':']:
            self.equivalent_char[c] = ' '
        self.equivalent_char['—'] = ''
        # This 'œ' in self.equivalent_char returns False... why ?
        # self.equivalent_char['œ'] = 'oe'
        self.equivalent_char['ç'] = 'c'
        self.equivalent_char['’'] = '\''

    @property
    def eoschar(self):
        return chr(self._EOS)

    @property
    def soschar(self):
        return chr(self._SOS)

    def encode(self, utterance):
        utterance = utterance.lower()
        # Remove the accentuated characters
        utterance = [self.equivalent_char[c] if c in self.equivalent_char else c for c in utterance]
        # Replace the unknown characters
        utterance = ['❌' if c not in self.char2idx else c for c in utterance]
        return [self.char2idx[c] for c in utterance]

    def decode(self, tokens):
        return "".join([self.idx2char[it] for it in tokens])


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
        self.charmap = CharMap()

    def __call__(self, batch):
        """
        Builds and return a minibatch of data as a tuple (inputs, targets)
        All the elements are padded to be of equal time

        Returns:
            a tuple (spectros, targets) with :
                spectors : (Batch size, n_mels, time)
                targets : (Batch size, time)
        """
        # Extract the subcomponents
        waveforms = [w for w, _, _ in batch]
        rates = set([r for _, r, _ in batch])
        transcripts = [
            torch.LongTensor(
                self.charmap.encode(self.charmap.soschar +
                                    d['sentence'] +
                                    self.charmap.eoschar)
            )
            for _, _, d in batch]
        # Retrieve the maximal length in the list of waveforms
        # max_len = max([w.shape[1] for w in waveforms])

        # Pad and stack the variable length tensors
        # waveforms is a 2D tensor (Batch, Time)
        waveforms = pad_sequence([t.squeeze() for t in waveforms],
                                 batch_first=True)

        spectrograms = self.transform(waveforms)

        transcripts = pad_sequence(transcripts, batch_first=True)

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
    dataset_loader = functools.partial(load_dataset,
                                       commonvoice_root=commonvoice_root)
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

if __name__ == '__main__':
    # Data loading
    train_loader, valid_loader, test_loader = get_dataloaders(_DEFAULT_COMMONVOICE_ROOT,
                                                              n_threads=4,
                                                             batch_size=10)

    X, y = next(iter(train_loader))
    print(X.shape)
    print(y)
    charmap = CharMap()
    for yi in y:
        print(charmap.decode(yi))

    print(charmap.decode(charmap.encode("nous sommes heureux de vous souhaiter nos meilleurs vœux pour 2015")))
    print('œ' in charmap.char2idx)
