#!/usr/bin/env python3
# coding: utf-8

import os
from typing import Optional, Any

import tensorflow as tf


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
