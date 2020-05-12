#!/usr/bin/env python3
# coding: utf-8

from argparse import Namespace
from typing import List, Tuple


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
