#!/usr/bin/env python3

# Standard import
import argparse
import json
# External import
import matplotlib.pyplot as plt


def plot_stats(args):
    with open(args.statfile, 'r') as f:
        stats = json.load(f)

    rate = stats['waveform']['sample_rates'][0]
    if len(stats['waveform']['sample_rates']) > 1:
        print("Multiple sampling rates, I take the first one of {}".format(stats['waveform']['sample_rates']))
    num_samples = list(stats['waveform']['num_samples'].items())
    num_samples = sorted(num_samples, key=lambda tu: tu[0], reverse=True)
    # Convert num_samples to second
    x_num_samples = [int(s)/rate for s, _ in num_samples]
    weights_num_samples = [num for _, num in num_samples]

    plt.figure()
    plt.hist(x_num_samples, weights=weights_num_samples)
    plt.xlabel(r"Time ($s$)")
    plt.title(f"Sample length from {args.statfile}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('statfile',
                        help="The stat file to process",
                        type=str)

    args = parser.parse_args()

    plot_stats(args)
