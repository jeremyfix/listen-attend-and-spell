#!/usr/bin/env python3

# Standard imports
import os
import collections
import json
# External imports
import torch.nn as nn
import torch.utils
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram
import tqdm
from nltk.tokenize import word_tokenize


# Download the tokenizer
import nltk
nltk.download('punkt')

_COMMONVOICE_ROOT="/opt/Datasets/CommonVoice"
_COMMONVOICE_FOLDER='CommonVoice'
_COMMONVOICE_VERSION="fr_79h_2019-02-25"

# Statistics
# Train

def compute_stats(dataset):
    stats = {'waveform':
             {
                 'sample_rates': set(),
                 'num_samples': collections.defaultdict(lambda: 0),
                 'cum_time': 0 # in s.
             },
             'transcripts':
             {
                 'lengths': collections.defaultdict(lambda: 0),
                 'vocabulary': collections.defaultdict(lambda: 0),
                 'vocab_size': 0

             }
            }
    for waveform, rate, dictionnary in tqdm.tqdm(dataset):
        stats['waveform']['sample_rates'].add(rate)
        stats['waveform']['num_samples'][waveform.shape[1]] += 1
        stats['waveform']['cum_time'] += waveform.shape[1]/rate
        # Remove the dash for words like quatre-vingt-dix
        # and remove the capital letters
        transcript = dictionnary['sentence'].replace('-', ' ').lower()
        # Tokenize the transcript
        transcript = word_tokenize(transcript)
        stats['transcripts']['lengths'][len(transcript)] += 1
        for v in transcript:
            stats['transcripts']['vocabulary'][v] += 1
    stats['transcripts']['vocab_size'] = len(stats['transcripts']['vocabulary'])
    # Convert the sets to lists if any (basically, this allows the dictionnary
    # to be serialized with json)
    stats['waveform']['sample_rates'] = list(stats['waveform']['sample_rates'])
    # Sort the vocabulary 
    vocab = stats['transcripts']['vocabulary']
    stats['transcripts']['vocabulary'] = sorted(vocab.items(),
                                                key=lambda tu:tu[1],
                                                reverse=True)
    return stats

def load_dataset(fold):
    datapath = os.path.join(_COMMONVOICE_ROOT,
                           _COMMONVOICE_VERSION, 'fr')
    return COMMONVOICE(root=datapath,
                       tsv=fold+'.tsv')

def test001():
    sets = ['test', 'train', 'dev']
    for s in sets:
        dataset = load_dataset(s)
        stats = compute_stats(dataset)
        with open('stats_{}.json'.format(s), 'w') as f:
            json.dump(stats, f, ensure_ascii=False)

class TrimZeroPad(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        """
        batch is list of size batch_size

        It contains tuples (waveform, sample_rate, dictionnary)
        """
        # Extract the subcomponents
        waveforms = [w for w, _, _ in batch]
        rates = set([r for _, r, _ in batch])
        transcripts = [d['sentence'] for _, _, d in batch]
        # Retrieve the maximal length in the list of waveforms
        max_len = max([w.shape[1] for w in waveforms])

        # Pad and stack the variable length tensors
        # waveforms is a 2D tensor (Batch, Time)
        waveforms = pad_sequence([t.squeeze() for t in waveforms],
                                 batch_first=True)
        if len(rates) != 1:
            raise NotImplementedError("Cannot deal with more than 1 sample rate in the data")

        return waveforms, rates.pop(), transcripts


def test002():
    import numpy as np
    import matplotlib.pyplot as plt
    batch_size = 4
    n_threads = 4
    dataset = load_dataset('train')
    trim_zero_pad = TrimZeroPad()
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=n_threads,
                                         collate_fn=trim_zero_pad)

    waveforms, rate, transcripts = next(iter(loader))

    win_length = 25 # ms
    win_step = 15 # ms
    nfft = int(win_length*1e-3*rate)
    noverlap = int(win_step * 1e-3 * rate)

    transform = nn.Sequential(
        Spectrogram(n_fft=nfft, hop_length = noverlap),
        MelScale(),
        AmplitudeToDB()
        # MelSpectrogram(sample_rate=rate, 
        #               n_fft=nfft,
        #               hop_length=noverlap)
    )
    spectro = transform(waveforms)

    fig, axes = plt.subplots(nrows=batch_size, ncols=1)
    for iax, (ax, spectroi) in enumerate(zip(axes, spectro)):
        # ax.specgram(waveforms[iax].numpy(), nfft, rate, noverlap=noverlap)

        #TODO: not completly sure about the time x-axis extent
        ax.imshow(spectroi,
                  extent=[0, spectroi.shape[0]*win_step*1e-3,0, spectro.max()],
                  aspect='auto',
                 origin='lower')
        ax.set_title('{}'.format(transcripts[iax]))
        ax.set_label("Time (s.)")
    plt.tight_layout()
    plt.savefig('specro.png')
    plt.show()





if __name__ == '__main__':
    # test001()
    test002()
