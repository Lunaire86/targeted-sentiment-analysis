#!/usr/bin/env python3
# coding: utf-8

import logging
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from logging import Logger
from typing import List, Tuple, Optional


def get_filenames(args: Namespace) -> Tuple[str, str]:
    """String together (ha!) filenames for model and logger."""
    strings: List[str] = [
        args.name,
        'sg' if args.sg else 'cbow',
        f'{args.dim}',
        'bin'
    ]
    model_name = '.'.join(strings)
    log_name = '-'.join(strings[:-1])

    if args.zip:
        model_name = log_name
    return model_name, f'{log_name}.log'


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
            '--source',
            action='store',
            type=str,
            help='A directory containing files the model will train on.'
                 'The required format is one sentence per line, with '
                 'all necessary pre-processing carried out already.'
        )
        parser.add_argument(
            '--target',
            action='store',
            type=str,
            help='The directory in which to save the model.'
        )
        parser.add_argument(
            '--name', '-n',
            action='store',
            type=str,
            default='fastText',
            help='This will become part of the filename.'
        )
        parser.add_argument(
            '--dim',
            action='store',
            type=int,
            default=100,
            help='Dimensionality of the word vectors.'
        )
        parser.add_argument(
            '--window_size',
            action='store',
            type=int,
            default=5,
            help='The maximum distance between the current '
                 'and predicted word within a sentence.'
        )
        parser.add_argument(
            '--min_count',
            action='store',
            type=int,
            default=5,
            help='The model ignores all words with '
                 'total frequency lower than this.'
        )
        parser.add_argument(
            '--epochs',
            action='store',
            type=int,
            default=5,
            help='Number of training epochs.'
        )
        parser.add_argument(
            '--cbow',
            action='store_true',
            help='Train using the fastText Continuous Bag-of-Words algorithm.'
        )
        parser.add_argument(
            '--sg',
            action='store_true',
            help='Train using the fastText Skipgram algorithm.'
        )
        parser.add_argument(
            '--zip',
            action='store_true',
            help='Toggle this to save both model and vectors in a '
                 'compressed archive rather than only the model itself.'
        )
