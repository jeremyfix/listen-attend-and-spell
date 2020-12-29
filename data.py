#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import functools
from pathlib import Path
from typing import Union
# External imports
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.utils.data
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram

_DEFAULT_COMMONVOICE_ROOT = "/opt/Datasets/CommonVoice/"
_DEFAULT_COMMONVOICE_VERSION = "v1"
_DEFAULT_RATE = 48000  # Hz
_DEFAULT_WIN_LENGTH = 25  # ms
_DEFAULT_WIN_STEP = 15  # ms
_DEFAULT_NUM_MELS = 40


def load_dataset(fold: str,
                 commonvoice_root: Union[str, Path],
                 commonvoice_version: str) -> torch.utils.data.Dataset:
    """
    Load the commonvoice dataset

    Args:
        fold (str): the fold to load, e.g. train, dev, test, validated, ..

    Returns:
        torch.utils.data.Dataset: ``dataset``
    """
    datasetpath = os.path.join(commonvoice_root, commonvoice_version, 'fr')
    return COMMONVOICE(root=datasetpath,
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
    def vocab_size(self):
        return len(self.idx2char)

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


class WaveformProcessor(object):

    def __init__(self):
        nfft = int(_DEFAULT_WIN_LENGTH * 1e-3 * _DEFAULT_RATE)
        # We need to memorize nstep since it is the downscaling
        # factor from the waveform to the spectrogram
        self.nstep = int(_DEFAULT_WIN_STEP * 1e-3 * _DEFAULT_RATE)
        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=_DEFAULT_RATE,
                           n_fft=nfft,
                           hop_length=self.nstep,
                           n_mels=_DEFAULT_NUM_MELS),
            AmplitudeToDB()
        )

    def get_spectro_length(self, waveform_length: int):
        """
        Computes the length of the spectrogram given the length
        of the waveform
        """
        return waveform_length//self.nstep+1

    def __call__(self, waveforms: torch.Tensor):
        """
        Apply the transformation on the input waveform tensor
        The time dimension is smalled because of the hop_length given
        to the MelSpectrogram object.

        Args:
            waveforms(torch.Tensor) : (B, Tx) waveform
        Returns:
            spectrograms(torch.Tensor): (B, Tx//nstep + 1, n_mels)
        """
        # spectrograms is (B, n_mel, Tx)
        # we permute it to be (B, Tx, n_mel)
        return self.transform(waveforms).permute(0, 2, 1)


class BatchCollate(object):
    """
    Collator for the individual data to build up the minibatches
    """

    def __init__(self):
        self.waveform_processor = WaveformProcessor()
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

        # Sort the waveforms and transcripts by decreasing waveforms length
        wt_sorted = sorted(zip(waveforms, transcripts),
                           key=lambda wr: wr[0].shape[1],
                           reverse=True)
        waveforms = [wt[0] for wt in wt_sorted]
        transcripts = [wt[1] for wt in wt_sorted]

        # Compute the lenghts of the spectrograms from the lengths
        # of the waveforms
        waveforms_lengths = [w.shape[1] for w in waveforms]
        spectro_lengths = [self.waveform_processor.get_spectro_length(wl) for wl in waveforms_lengths]
        transcripts_lengths = [t.shape[0] for t in transcripts]

        # Pad the waveforms to the longest waveform
        # so that we can process them as a batch through the transform
        waveforms = pad_sequence([t.squeeze() for t in waveforms],
                                 batch_first=True)

        spectrograms = self.waveform_processor(waveforms)
        spectrograms = pack_padded_sequence(spectrograms,
                                            lengths=spectro_lengths,
                                            batch_first=True)

        # transcripts is (B, Ty)
        transcripts = pad_sequence(transcripts,
                                   batch_first=True)
        transcripts = pack_padded_sequence(transcripts,
                                           lengths=transcripts_lengths,
                                           enforce_sorted=False,
                                           batch_first=True)

        if len(rates) != 1:
            raise NotImplementedError("Cannot deal with more than 1 sample rate in the data")
        if rates.pop() != _DEFAULT_RATE:
            raise NotImplementedError("One batch is using a sampling rate different"
                                      f" from the assumed {self._DEFAULT_RATE} Hz")
        return spectrograms, transcripts


def get_dataloaders(commonvoice_root: str,
                    commonvoice_version: str,
                    cuda: bool,
                    batch_size: int = 64,
                    n_threads: int = 4,
                    small_experiment:bool = False):
    """
    Build and return the pytorch dataloaders

    Args:
        commonvoice_root (str or Path) : the root directory where the dataset
                                         is stored
        commonvoice_version (str) : the version of the dataset to consider, e.g. 1, 6.1, ..
        cuda (bool) : whether to use cuda or not. Used for creating tensors
                      on the right device
        batch_size (int) : the number of samples per minibatch
        n_threads (int) : the number of threads to use for dataloading
        small (bool) : whether or not to use small subsets, usefull for debug
    """
    dataset_loader = functools.partial(load_dataset,
                                       commonvoice_root=commonvoice_root,
                                       commonvoice_version=commonvoice_version)
    train_dataset = dataset_loader("train")
    valid_dataset = dataset_loader("dev")
    test_dataset = dataset_loader("test")
    if small_experiment:
        indices = range(3*batch_size)
        train_dataset = torch.utils.data.Subset(train_dataset,
                                                indices=indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset,
                                                indices=indices)
        test_dataset = torch.utils.data.Subset(test_dataset,
                                               indices=indices)

    batch_collate_fn = BatchCollate()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=n_threads,
                                               collate_fn=batch_collate_fn,
                                               pin_memory=cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=n_threads,
                                               collate_fn=batch_collate_fn,
                                               pin_memory=cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=n_threads,
                                              collate_fn=batch_collate_fn,
                                              pin_memory=cuda)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    # Data loading
    train_loader, valid_loader, test_loader = get_dataloaders(_DEFAULT_COMMONVOICE_ROOT,
                                                              _DEFAULT_COMMONVOICE_VERSION,
                                                              cuda=False,
                                                              n_threads=4,
                                                              batch_size=10)

    X, y = next(iter(train_loader))
    X, lens_X = pad_packed_sequence(X, batch_first=True)
    y, lens_y = pad_packed_sequence(y, batch_first=True)
    print(X.shape)
    charmap = CharMap()
    for yi, li in zip(y, lens_y):
        print(charmap.decode(yi)[:li])

    print(charmap.decode(charmap.encode("nous sommes heureux de vous souhaiter nos meilleurs vœux pour 2015")))
    print('œ' in charmap.char2idx)

    print(f"The vocabulary contains {charmap.vocab_size} characters")
