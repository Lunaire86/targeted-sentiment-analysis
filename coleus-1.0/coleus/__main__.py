#!/usr/bin/env python3
# coding: utf-8

from time import time
from logging import Logger
from argparse import Namespace
from os import path
from zipfile import ZipFile

from src.model import Model
from src.utils import get_filenames, ArgParser, get_logger, log_msg

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
    logger = None if args.quiet else get_logger(log_file)
    log_msg(f'Program arguments:\n{vars(args)}', logger)
    print(f'Log file: {log_file if logger else None}')

    # Initialise (build) the model.
    model = Model.init(args)

    # If the model should take in additional data, do
    # model.build_vocab(more_training_data, update=True)
    # before moving on to the training.

    log_msg('Training...')
    t = time()
    model.train(args.source)
    training_time = time() - t

    log_msg(f'Finished training in {training_time} seconds (~{int(training_time / 60)} minute(s)).')
    log_msg(model.__str__())
    print(model)

    mode = 'wb' if path.exists(model_path) else 'xb'
    if args.zip:
        with ZipFile(model_path, mode[0]) as archive:
            archive.write(filename=model.save(), arcname='model.bin')
            archive.write(filename=model.wv, arcname='model.vec')
            log_msg(f'Saved to archive {model_path} -> model.bin, model.vec')
    else:
        with open(model_path, mode) as f:
            model.save(f)
        log_msg(f'Saved to file -> {model_path}')

