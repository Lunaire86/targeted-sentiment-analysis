#!/usr/bin/env python3
# coding: utf-8

import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, InitVar

# These informational messages get printed to
# the terminal before any of the input prompts.
user_info = [
    'PURPOSE: set project level variables for directories '
    'containing i.e. word embeddings, datasets and models.\n',
    'REASON: allow for the loading of pickled versions, if '
    'they exist, without it having to be explicitly specified.\n'
    'Setting these variables here takes away the need for having '
    'them passed in as command line arguments each time.\n',

    'Some user errors will have been accounted for, but probably '
    'not all. Please keep this in mind when passing input to this script.\n',

    'User input like yes and no is case-insensitive and can be shortened '
    'to a single letter. If the expected input is a path to a file or '
    'directory, it is case-sensitive. The user is not required to surround '
    'these paths by single or double quotes, but it is encouraged.\n'
]

# Probably not the most elegant solution, but it sort of works
input_prompts = {
    'project_root': f'\n{"=" * 22}\nProject root directory\n{"=" * 22}\n'
                    f'Is {os.getcwd()} the project root directory?\n'
                    f'Yes  -> [ENTER]\n'
                    f'No   -> specify path',

    'word_embeddings': f'\n{"=" * 27}\nPre-trained word embeddings\n{"=" * 27}\n'
                       f'Where should the program look for word embeddings?\n'
                       f'project_root/models    -> [ENTER]\n'
                       f'Saga shared directory  -> [S]\n'
                       f'Other                  -> specify path\n'
}


@dataclass
class PathTo:
    project_root: str = os.getcwd()
    saga_shared: str = '/cluster/shared/nlpl/data/vectors/latest'
    embeddings: str = field(init=False)

    # Datasets
    data: str = field(init=False)
    raw_data: str = field(init=False)
    interim_data: str = field(init=False)
    processed_data: str = field(init=False)
    train: str = field(init=False)
    dev: str = field(init=False)
    test: str = field(init=False)

    SETUP: InitVar[bool] = False

    def __post_init__(self, SETUP):
        if SETUP:
            self._run_setup()
        else:
            self.embeddings = os.path.join(self.project_root, 'models')

        # Datasets
        self.data = os.path.join(self.project_root, 'data')
        self.raw_data = os.path.join(self.data, 'raw')
        self.interim_data = os.path.join(self.data, 'interim')
        self.processed_data = os.path.join(self.data, 'processed')
        self.train, self.dev, self.test = [
            os.path.join(self.raw_data, basename)
            for basename in ('train', 'dev', 'test')
        ]

    def _run_setup(self):
        for msg in user_info:
            print(msg)
            time.sleep(0.15)

        # Set the different paths
        self.project_root = self._set_path(input(input_prompts['project_root']))
        self.embeddings = self._set_path(input(input_prompts['word_embeddings']))

    def _get_abspath(self, path):
        """Returns an absolute path."""
        abspath = os.path.expanduser(  # expand ~/
            os.path.expandvars(  # expand any shell variables, i.e. $SOME_PATH
                os.path.abspath(  # get absolute (normalised) path
                    path)))  # path from user
        return abspath

    def _set_path(self, user_input):
        abspath = self._get_abspath(user_input)

        if os.path.isdir(abspath):
            return abspath

        raise NotADirectoryError(f"User input: {user_input}\n"
                                 f"Expanded path: {abspath}")


@dataclass
class ArgParser:
    parser: ArgumentParser = field(init=False, default_factory=ArgumentParser)
    args: Namespace = field(init=False)

    def __post_init__(self):
        self._add_args()
        self.args = self.parser.parse_args()

    def _add_args(self):
        parser = self.parser
        parser.add_argument(
            '--num_layers', '-nl',
            action='store',
            type=int,
            default=1,
            help='Number of layer(s)'
        )
        parser.add_argument(
            '--hidden_dim', '-hd',
            action='store',
            type=int,
            default=100,
            help='Size of the hidden layer(s)'
        )
        parser.add_argument(
            '--batch_size', '-bs',
            action='store',
            type=int,
            default=50,
            help='Size of mini-batches'
        )
        parser.add_argument(
            '--dropout', '-dr',
            action='store',
            type=float,
            default=0.01
        )
        parser.add_argument(
            '--learning_rate', '-lr',
            action='store',
            type=float,
            default=1e-3,
            help='Learning rate'
        )
        parser.add_argument(
            '--epochs', '-e',
            action='store',
            type=int,
            default=50,
            help='Early stopping is implemented, so this may be an arbitrary large number'
        )
        parser.add_argument(
            '--embedding_dim', '-edim',
            action='store',
            type=int,
            default=100,
            help='Dimensionality of the word embeddings'
        )
        parser.add_argument(
            '--train_embeddings', '-tre',
            action='store_true',
            help='Trainable embeddings?'
        )
        parser.add_argument(
            '--embedding_id', '-eid',
            action='store',
            type=int,
            default=58,
            help='Word embedding ID number (as per http://vectors.nlpl.eu)'
        )
        parser.add_argument(
            '--embeddings_dir', '-edir',
            action='store',
            type=str,
            default='/cluster/shared/nlpl/vectors/latest',
            help='Path to word embeddings root directory'
        )
        parser.add_argument(
            '--saga',
            action='store_true',
            help='Toggle if running on Saga'
        )
