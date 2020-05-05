#!/usr/bin/env python3
# coding: utf-8

import os
import pickle
import time
from collections import Counter, OrderedDict
from os.path import join
from typing import List, Any

# plotting
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from data.preprocessing import Dataset
from utils.config import PathTracker, set_global_seed


def setup():
    sns.set()
    sns.set_context('poster', font_scale=1.3)
    sns.set_style("white")

    # Update matplotlib defaults to something nicer
    mpl_update = {
        'font.size': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': [12.0, 8.0],
        'axes.labelsize': 20,
        'axes.labelcolor': '#677385',
        'axes.titlesize': 20,
        'lines.color': '#0055A7',
        'lines.linewidth': 3,
        'text.color': '#677385',
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': 'Tahoma'
    }
    mpl.rcParams.update(mpl_update)
    mpl.use('Qt5Agg')


# def plot_histogram(sequences: List[List[Any]], bins=10):
#
#     sizes = [len(_) for _ in sequences]
#     hist_array = np.histogram(sizes, bins=bins)
#
#     plt.hist(hist_array)
#     plt.title('Sequence length histogram')
#     plt.savefig(os.path.join(path_to.project_root, 'figures', name))
#     plt.show()


def plot_freqs(counter, name):
    """Plot a bar chart showing the word length frequency."""
    t = time.strftime('%m%d_%H-%M-%S')
    img_name = f'{t}_{name}_seq_len_freq.png'

    len_, freq = zip(*counter.items())

    # sort your values in descending order
    # Returns the indices that would sort an array:
    # Perform an indirect sort along the given axis using the algorithm
    # specified by the kind keyword. It returns an array of indices of the
    # same shape as a that index data along the given axis in sorted order.
    ind_sort = np.argsort(freq)[::-1]
    len_ = np.array(len_)[ind_sort]
    freq = np.array(freq)[ind_sort]

    indexes = np.arange(len(len_))

    bar_width = 0.35
    low, high = min(counter.keys()), max(counter.keys())
    high += 10 - (high % 10)  # round up to the nearest tenner
    x_ticks = np.arange(low - 1, high, 10)
    x_ticks[0] = 1

    plt.bar(indexes, freq)
    plt.title(f'Sentence length by frequency: {name}')
    plt.xlabel('Sentence length')
    plt.ylabel('Frequency')
    plt.xticks(x_ticks)
    plt.xlim((low - 2, high))
    plt.show()
    plt.savefig(os.path.join(path_to.project_root, 'figures', img_name))
    plt.close()


def pretty_print(frequency, bins):
    for b, f in zip(bins[1:], frequency):
        print(round(b, 1), ' '.join(np.repeat('*', f)))


def relative_length(c: Counter, percentile: int):
    tuples = []
    for k, v in c.items():
        if k > max(c.keys()) * percentile / 100:
            tuples.append((k, v))

    tuples.sort(key=lambda x: x[0])
    od = OrderedDict(tuples)

    print(f'Sentence length > percentile {percentile}\n'
          f'Sentence count: {sum(od.values())}/{sum(c.values())}\n'
          f'Unique lengths: {len(od)}')
    print(od.items(), '\n')


def write_to_csv(filepath, data):
    raw_count = [len(_) for _ in data]
    raw_count.sort()
    freqs = Counter(raw_count)
    s = [f'{c},{freqs[c]}\n' for c in raw_count]
    with open(filepath, 'w') as f:
        f.write('length,frequency\n')
        f.writelines(s)


if __name__ == '__main__':
    setup()
    set_global_seed()
    print('Global seed set!')

    path_to = PathTracker.from_json(
        join(os.pardir, os.pardir, 'local_config.json')
    )

    print('Reading data from conll...')
    train = Dataset(path_to.train)
    dev = Dataset(path_to.dev)
    print('Done!')

    train_counter = Counter([len(_) for _ in train.X])
    dev_counter = Counter([len(_) for _ in dev.X])


    # write_to_csv(filepath=join(path_to.figures, 'train_sent_len.csv'), data=train.X)
    # write_to_csv(filepath=join(path_to.figures, 'dev_sent_len.csv'), data=dev.X)


    print(f'\nSentence count\ntrain: {len(train.X)}\ndev:   {len(dev.X)}\n'
          f'\nShortest sentence\ntrain: {min(train_counter.keys())}\ndev:   {min(dev_counter.keys())}\n'
          f'\nLongest sentence\ntrain: {max(train_counter.keys())}\ndev:   {max(dev_counter.keys())}\n')

    print(f'\n{"~" * 5}\nTRAIN\n{"~" * 5}')
    for p in np.arange(50, 75, 5):
        # relative_length(train_counter, p)
        pass
    print(f'\n{"~" * 3}\nDEV\n{"~" * 3}')
    for p in np.arange(50, 75, 5):
        # relative_length(dev_counter, p)
        pass

    # plot_freqs(c, 'train')


