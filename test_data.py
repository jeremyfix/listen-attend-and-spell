#!/usr/bin/env python3

# Standard imports
import collections
import json
# External imports
import torchaudio
from torchaudio.datasets import COMMONVOICE
import tqdm

_COMMONVOICE_ROOT="/opt/Datasets/"
_COMMONVOICE_VERSION="fr_79h_2019-02-25"

# Statistics
# Train

def compute_stats(dataset):
    stats = {'waveform':
             {
                 'sample_rates': set(),
                 'num_samples': collections.defaultdict(lambda: 0)
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
        transcript = dictionnary['sentence']
        stats['transcripts']['lengths'][len(transcript.split(' '))] += 1
        for v in transcript.split():
            stats['transcripts']['vocabulary'][v] += 1
    stats['transcripts']['vocab_size'] = len(stats['transcripts']['vocabulary'])
    return stats


def test001():
    def load_dataset(fold):
        return COMMONVOICE(root=_COMMONVOICE_ROOT,
                           version=_COMMONVOICE_VERSION,
                           tsv=fold+'.tsv',
                           url='french')
    sets = ['train', 'test', 'dev']
    for s in sets:
        dataset = load_dataset(s)
        stats = compute_stats(dataset)
        with open('stats_{}.json'.format(s), 'w') as f:
            json.dump(stats, f)





if __name__ == '__main__':
    test001()
