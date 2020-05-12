#!/usr/bin/env python3
# coding: utf-8

import logging
import os
from time import time
from zipfile import ZipFile

from coleus.config import ArgParser
from coleus.helpers import get_filenames
from coleus.model import FastTextModel
from utils.config import PathTracker


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('reading')


class NoThreadsFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('worker')


if __name__ == '__main__':
    # Store command line arguments
    parser = ArgParser()  # class defined in coleus/config.py
    args = parser.args
    print(args, sep='\n')

    # Easy access to various file locations
    path_to = (
        PathTracker.from_json('saga_config.json') if args.saga
        else PathTracker.from_json('local_config.json')
    )

    # Get filenames for model and logger.
    model_file, log_file = get_filenames(args)
    model_path = (
        os.path.join(args.target, model_file)
        if args.target else path_to.embeddings
    )

    # Create a logger and print the path to the log file
    logger = logging.getLogger()
    handler = logging.FileHandler(
        filename=os.path.join(path_to.logs, log_file),
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
    model = FastTextModel.init(args)

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

    mode = 'wb' if os.path.exists(model_path) else 'xb'
    if args.zip:
        with ZipFile(model_path, mode[0]) as archive:
            archive.write(filename=model.save(), arcname='model.bin')
            archive.write(filename=model.wv, arcname='model.vec')
            logger.info(f'Saved to archive {model_path} -> model.bin, model.vec')
    else:
        with open(os.path.join(model_path, model_file), mode) as f:
            model.save(f)
        logger.info(f'Saved to file -> {model_path}')
