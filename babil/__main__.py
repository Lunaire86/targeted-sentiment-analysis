#!/usr/bin/env python3
# coding: utf-8

import logging

import matplotlib as mpl

import coleus
from models import run
from utils.config import ArgParser, PathTracker, set_global_seed

if __name__ == '__main__':
    # Store command line arguments
    parser = ArgParser()  # class defined in babil/utils/config.py
    args = parser.args
    print(args, sep='\n')

    # To avoid issues
    mpl.use('Agg')

    # Initialise a logger
    logging.basicConfig(filename='babil.log', filemode='a',
                        format='{asctime} : {levelname} : {message}',
                        datefmt='%Y-%m-%d %X',
                        style='{',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Ensure reproducibility
    set_global_seed()

    # Easy access to various file locations
    paths = (
        PathTracker.from_json('saga_config.json') if args.saga
        else PathTracker.from_json('local_config.json')
    )

    s = '~' * len(args.run)
    if args.run == 'baseline':
        print(f'{s}\nBASELINE\n{s}')
        logger.info(f'\n{"~" * 40} NEW JOB - BASELINE {"~" * 40}\n')
        run.baseline_trainer(args, paths, logger)

    elif args.run == 'improved':
        print(f'{s}\nIMPROVED\n{s}')
        logger.info(f'\n{"~" * 40} NEW JOB - IMPROVED {"~" * 40}\n')
        run.improved_trainer(args, paths, logger)


