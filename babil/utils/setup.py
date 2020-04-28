#!/usr/bin/env python3
# coding: utf-8

import os
import time
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
    embeddings_root: str = field(init=False)
    # pickle_root: str = field(init=False)  # probably not needed
    SETUP: InitVar[bool] = False

    def __post_init__(self, SETUP):
        if SETUP:
            self._run_setup()

    def _run_setup(self):
        for msg in user_info:
            print(msg)
            time.sleep(0.15)

        # Set the different paths
        self._set_path(input(input_prompts['project_root']))
        self._set_path(input(input_prompts['word_embeddings']))

    def _set_path(self, user_input):
        new_path = ''
        # TODO : implement
        try:
            # Check if user has given a valid path
            # os.path.join(os.getcwd(), 'models') .. ?
            pass
        except FileNotFoundError:
            raise FileNotFoundError("That's a no from me.")
        # Check if generated path is valid
        #     return if true; raise if false


if __name__ == '__main__':
    # Default paths
    project_root = os.getcwd()
    embeddings_root = os.path.join(os.getcwd(), 'models')
    pickle_root = os.path.join(os.getcwd(), 'models')



    # Set project root

    prompt_1 = input(prompts[1])
    if not os.path.abspath(prompt_1) == project_root:
        project_root = os.path.abspath(prompt_1) if prompt_1 else os.path.abspath(os.getcwd())

    # Store path to pre-trained word embeddings
    prompt_2 = input(prompts[2])
    if prompt_2 == 's'.casefold():
        embeddings_root = saga
    embeddings_root = os.path.abspath(prompt_2) if prompt_2 else project_root

    # Store path to other models
    prompt_3 = input('')

    # for dirpath, dirnames, filenames in os.walk(absroot):
    #     if not os.path.basename(dirpath).startswith('.'):
    #         print(dirnames)
