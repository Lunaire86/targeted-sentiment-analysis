#!/usr/bin/env python3
# coding: utf-8
import logging

from babil.models import baseline
from babil.utils.config import ArgParser, PathTracker, set_global_seed

if __name__ == '__main__':
    # Store command line arguments
    parser = ArgParser()  # class defined in babil/utils/config.py
    args = parser.args
    print(args, sep='\n')

    # Initialise a logger
    logging.basicConfig(filename='logger.log', filemode='a',
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
    logger.info(f'Project paths set:\n{paths}')

    if args.run == 'baseline':
        s = '~' * len(args.run)
        print(f'{s}\nBASELINE\n{s}')
        baseline.run(args, paths, logger)
