#!/usr/bin/env python3
# coding: utf-8

import time
from argparse import ArgumentParser
from os import path
from os.path import join

from multiprocessing import cpu_count

import numpy as np
from gensim.models.callbacks import CallbackAny2Vec

from gensim.models.fasttext import FastText
from gensim.models.word2vec import PathLineSentences


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0
        self.timer = 0

    def on_epoch_begin(self, model):
        t = time.strftime('%H-%M-%S')
        self.timer = time.time()
        print(f'{t} : Epoch #{self.epoch} start')

    def on_epoch_end(self, model):
        t = time.strftime('%H-%M-%S')
        print(f'{t} : Epoch #{self.epoch} end\n'
              f'Duration -> {round(time.time() - self.timer, 2)} seconds\n')
        self.epoch += 1
        self.timer = 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--train',
        action='store',
        type=str,
        help='Path to a directory containing .cor (or other text) files'
    )
    parser.add_argument(
        '--save',
        action='store',
        type=str,
        help='Where to save the embeddings'
    )
    parser.add_argument(
        '--dim',
        action='store',
        type=int,
        default=300,
        help='Embedding dimensions'
    )
    parser.add_argument(
        '--window_size',
        action='store',
        type=int,
        default=5,
        help='Size of the word window'
    )
    parser.add_argument(
        '--epochs',
        action='store',
        type=int,
        default=10,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--saga',
        action='store_true',
        help='Toggle if running on Saga'
    )
    args = parser.parse_args()

    model: FastText
    corpus = PathLineSentences(args.train)

    partition = path.basename(args.train)
    model_name = f'norec.{partition}.{args.dim}.bin'

    epoch_logger = EpochLogger()
    num_cores = cpu_count()

    if args.saga:
        print('\n\n', args)
        print(f'Model is being trained on {partition}\n'
              f'Model is being saved as {model_name}\n')
        print(f'Number of reviews to train on: '
              f'{len(corpus.input_files)}')

        model = FastText(
            # Path to a corpus file in LineSentence format
            PathLineSentences(args.train),

            # Dimensionality of the word vectors.
            size=args.dim,

            # The maximum distance between the current
            # and predicted word within a sentence.
            window=args.window_size,

            # The model ignores all words with total
            # frequency lower than this.
            min_count=2,

            # Training algorithm: skip-gram if `sg=1`,
            # otherwise CBOW.
            sg=1,

            # Number of iterations (epochs) over the corpus.
            iter=args.epochs,

            # Use these many worker threads to train the model
            # (=faster training with multicore machines).
            workers=int(num_cores / 2),

            # sort the vocabulary by descending frequency
            # before assigning word indices
            sorted_vocab=1,
        )  # instantiate

        print('Training...')
        train_start = time.time()
        model.train(
            # Path to a corpus file in LineSentence format
            PathLineSentences(args.train),

            # Count of sentences.
            total_examples=model.corpus_count,

            # Count of raw words in sentences.
            total_words=model.corpus_total_words,

            # Count of words already trained. Set this to 0 for
            # the usual case of training on all words in sentences.
            word_count=0,

            # Number of iterations (epochs) over the corpus.
            epochs=model.epochs,

            # List of callbacks that will be executed/run
            # at specific stages during training.
            callbacks=[epoch_logger]
        )  # train

        training_time = time.time() - train_start
        print(f'Done!\n'
              f'Training took {training_time} seconds (~{int(training_time / 60)} minutes).')

    else:
        # Run locally to do small-scale testing
        sentences = []
        for file in corpus:
            sentences.append(file)

        model = FastText(
            # A list of sentences
            # sentences,

            # Dimensionality of the word vectors.
            size=args.dim,

            # The maximum distance between the current
            # and predicted word within a sentence.
            window=5,

            # The model ignores all words with total
            # frequency lower than this.
            min_count=1,

            # Training algorithm: skip-gram if `sg=1`,
            # otherwise CBOW.
            sg=1,

            # Number of iterations (epochs) over the corpus.
            iter=args.epochs,

            # Use these many worker threads to train the model
            # (=faster training with multicore machines).
            workers=4,

            # sort the vocabulary by descending frequency
            # before assigning word indices
            sorted_vocab=1,
        )  # instantiate

        model.build_vocab(sentences)
        model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )

        s = 'kjempejalla'
        print(f'{s} in vocab: {s in model.wv}')
        old_vector = np.copy(model.wv[s])
        print(f'{s} vector: {old_vector}')

        sents = [
            ['dette', 'er', 'en', 'kjempesnål', 'setning', '.'],
            ['hubbabubbaklubb', 'kjempejalla', 'oppvaskbørstesalsa'],
            ['jubaluba', 'wubba-lubba-dub-dub', 'jiiihaaaaaa']
        ]
        print('Test sentences: ')
        print(*sents, sep='\n')

        # Add new sentences
        print('Updating vocab with test sentences...')
        model.build_vocab(sents, update=True)
        model.train(sents, total_examples=len(sents), epochs=model.epochs)

        new_vector = np.copy(model.wv['kjempesnål'])
        print(f'kjempesnål in vocab: {"kjempesnål" in model.wv}')
        print(f'kjempesnål vector: {new_vector}')

        print(f'Vocab size: {len(model.wv.index2entity)}')

    # Stats
    print(f'Vocab size: {len(model.wv.index2entity)}')
    print(f'Checking word similarities:\n'
          f'morsom\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("morsom", topn=5)]}\n\n'
          f'kjedelig\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("kjedelig", topn=5)]}\n\n'
          f'spennende\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("spennende", topn=5)]}\n\n'
          f'innviklet\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("innviklet", topn=5)]}\n\n'
          f'best\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("best", topn=5)]}\n\n')

    if args.saga:
        target_dir = args.save
        model_path = join(target_dir, model_name)
        mode = 'wb' if path.exists(model_path) else 'xb'
        with open(model_path, mode) as f:
            model.save(f)
