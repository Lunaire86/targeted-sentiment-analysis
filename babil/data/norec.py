#!/usr/bin/env python3
# coding: utf-8

import time
from argparse import ArgumentParser
from os import path
from os.path import join

from multiprocessing import cpu_count

from gensim.models.fasttext import FastText
from gensim.models.word2vec import PathLineSentences


# class EpochLogger(CallbackAny2Vec):
#     '''Callback to log information about training'''
#
#     def __init__(self):
#         self.epoch = 0
#         self.timer = 0
#
#     def on_epoch_begin(self, model):
#         t = time.strftime('%H-%M-%S')
#         self.timer = time.time()
#         print(f'{t} : Epoch #{self.epoch} start')
#
#     def on_epoch_end(self, model):
#         t = time.strftime('%H-%M-%S')
#         print(f'{t} : Epoch #{self.epoch} end\n'
#               f'Duration -> {round(time.time() - self.timer, 2)} seconds\n')
#         self.epoch += 1
#         self.timer = 0


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
        default=100,
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
        default=5,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--cbow',
        action='store_true',
        help='fastText Continuous Bag-of-Words'
    )
    parser.add_argument(
        '--sg',
        action='store_true',
        help='fastText Skipgram'
    )

    args = parser.parse_args()

    model: FastText
    corpus = PathLineSentences(args.train)
    partition = path.basename(args.train)

    algorithm = 'sg' if args.sg else 'cbow'
    model_name = f'norec.{args.dim}.{algorithm}.bin'

    # epoch_logger = EpochLogger()
    num_cores = cpu_count()

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
        min_count=5,

        # Training algorithm: skip-gram if `sg=1`,
        # otherwise CBOW.
        sg=1 if args.sg else 0,

        # Number of iterations (epochs) over the corpus.
        iter=args.epochs,

        # Use these many worker threads to train the model
        # (=faster training with multicore machines).
        workers=min(32, num_cores),

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
        # callbacks=[epoch_logger]
    )  # train

    training_time = time.time() - train_start
    print(f'Done!\n'
          f'Training took {training_time} seconds (~{int(training_time / 60)} minutes).')

    target_dir = args.save
    model_path = join(target_dir, model_name)
    mode = 'wb' if path.exists(model_path) else 'xb'
    with open(model_path, mode) as f:
        model.save(f)

    # Stats
    print(model)
    print(f'Checking word similarities:\n'
          f'morsom\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("morsom", topn=5)]}\n\n'
          f'kjedelig\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("kjedelig", topn=5)]}\n\n'
          f'spennende\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("spennende", topn=5)]}\n\n'
          f'innviklet\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("innviklet", topn=5)]}\n\n'
          f'best\n{[(k, round(v, 4)) for (k, v) in model.wv.most_similar("best", topn=5)]}\n\n')
