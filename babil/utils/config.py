#!/usr/bin/env python3
# coding: utf-8

import json
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Optional, Any

import tensorflow as tf
from features import EMBEDDINGS


def _set_global_seed(seed_dir):
    global_seed: int
    seed_file = os.path.join(seed_dir, 'SEED')

    if os.path.exists(seed_file) and os.path.isfile(seed_file):
        try:
            with open(seed_file, 'r') as f:
                global_seed = int(f.read())
                return tf.random.set_seed(global_seed)

        except FileNotFoundError:
            raise FileNotFoundError("SEED file not found. Global seed not set!")
    else:
        return _set_global_seed(os.path.split(seed_dir)[0])


def set_global_seed() -> Optional[Any]:
    # Ensure reproducibility
    working_dir = os.path.abspath(os.path.curdir)
    return _set_global_seed(working_dir)


@dataclass
class PathTracker:
    project_root: str
    data: str
    raw_data: str
    interim_data: str
    processed_data: str
    shared_embeddings: str
    embeddings: str
    figures: str
    models: str
    logs: str
    train: str
    dev: str
    test: str

    @staticmethod
    def from_json(json_file):
        with open(json_file, 'r') as f:
            entries = json.load(f)
        return PathTracker(**entries)


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
            default=100,
            help='Size of mini-batches'
        )
        parser.add_argument(
            '--epochs', '-e',
            action='store',
            type=int,
            default=50,
            help='Early stopping is implemented, so this may be an arbitrary large number'
        )
        parser.add_argument(
            '--dropout', '-dr',
            action='store',
            type=float,
            default=0.01
        )
        parser.add_argument(
            '--recurrent_dropout', '-rdr',
            action='store',
            type=float,
            default=0.01
        )
        parser.add_argument(
            '--learning_rate', '-lr',
            action='store',
            type=float,
            default=0.01,
            help='Learning rate'
        )
        parser.add_argument(
            '--embeddings',
            action='store',
            type=str,
            choices=EMBEDDINGS,
            default='110.zip',
            help=str(EMBEDDINGS)
        )
        parser.add_argument(
            '--embeddings_dim', '-edim',
            action='store',
            type=int,
            default=100,
            help='Dimensionality of the pre-trained word embeddings'
        )
        parser.add_argument(
            '--train_embeddings', '-tre',
            action='store_true',
            help='Trainable embeddings?'
        )
        parser.add_argument(
            '--saga',
            action='store_true',
            help='Toggle if running on Saga'
        )
        parser.add_argument(
            '--run',
            action='store',
            type=str,
            choices=['baseline', 'improved'],
            default='baseline'
        )
        parser.add_argument(
            '--load',
            action='store_true',
            help='Toggle if loading checkpoint model'
        )
