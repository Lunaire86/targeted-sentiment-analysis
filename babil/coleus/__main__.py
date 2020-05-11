#!/usr/bin/env python3
# coding: utf-8

import logging
import os
from argparse import Namespace
from logging import Logger
from os import path
from time import time
from zipfile import ZipFile

from model import Model
from utils import get_filenames, ArgParser


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('reading')


class NoThreadsFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('worker')


if __name__ == '__main__':
    # Type declarations
    parser: ArgParser
    args: Namespace
    logger: Logger
    model: Model

    # Store command line arguments.
    parser = ArgParser()
    args = parser.args

    # Get filenames for model and logger.
    model_file, log_file = get_filenames(args)
    model_path = path.join(args.target, model_file)

    # Create a logger and print the path to the log file
    logger = logging.getLogger()
    handler = logging.FileHandler(
        filename=log_file,
    )
    formatter = logging.Formatter(
        fmt='{asctime} : {levelname} : {message}',
        datefmt='%Y-%m-%d %X',
        style='{'
    )
    handler.setFormatter(formatter)
    handler.addFilter(NoParsingFilter())
    handler.addFilter(NoThreadsFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(f'Program arguments:\n{vars(args)}')
    print(f'Log file: {os.path.abspath(log_file)}')

    # Initialise (build) the model.
    model = Model.init(args)

    # If the model should take in additional data, do
    # model.build_vocab(more_training_data, update=True)
    # before moving on to the training.

    logger.info('Training...')
    t = time()
    model.train(args.source)
    training_time = time() - t

    logger.info(f'Finished training in {training_time} seconds (~{int(training_time / 60)} minute(s)).')
    logger.info(model.__str__())
    print(model)

    mode = 'wb' if path.exists(model_path) else 'xb'
    if args.zip:
        with ZipFile(model_path, mode[0]) as archive:
            archive.write(filename=model.save(), arcname='model.bin')
            archive.write(filename=model.wv, arcname='model.vec')
            logger.info(f'Saved to archive {model_path} -> model.bin, model.vec')
    else:
        with open(model_path, mode) as f:
            model.save(f)
        logger.info(f'Saved to file -> {model_path}')

