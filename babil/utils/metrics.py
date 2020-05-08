#!/usr/bin/env python3
# coding: utf-8
from collections import namedtuple
from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, NamedTuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, classification_report

from utils.helpers import y_dict


@dataclass
class Metrics:
    # We're only using these to instantiate the class
    X: InitVar[np.ndarray]
    y_gold: InitVar[np.ndarray]
    y_pred: InitVar[np.ndarray]
    idx2label: InitVar[Dict[int, str]]

    # Create a dict that lets us access predictions
    # and targets in a number of different formats.
    __dict__: Dict[str, Dict[str, Dict[str, np.ndarray]]] = field(init=False, default_factory=dict)

    # Binary and proportional metrics stored as namedtuples
    binary: NamedTuple = field(init=False)
    regular: NamedTuple = field(init=False)
    samples: NamedTuple = field(init=False)

    # Multiclass confusion matrix
    cfm: np.ndarray = field(init=False)

    # Classification report
    report: str = field(init=False)

    def __post_init__(self, X, y_true, y_pred, idx2label):
        # Use InitVars to create internal dictionary
        self.__dict__ = create_dict(y_true, y_pred, X, idx2label)

        # Calculate binary scores (minority vs majority)
        self.binary = binary_scores(
            y_true=self.__dict__['gold']['flat']['binary'],
            y_pred=self.__dict__['pred']['flat']['binary']
        )

        # Calculate proportional scores amongst minority classes
        self.regular = proportional_scores(
            y_true=self.__dict__['gold']['flat']['vectorised'],
            y_pred=self.__dict__['pred']['flat']['vectorised'],
            idx2label=idx2label
        )

        # Calculate metrics for each class, and find their average
        self.samples = proportional_scores(
            y_true=self.__dict__['gold']['flat']['vectorised'],
            y_pred=self.__dict__['pred']['flat']['vectorised'],
            idx2label=idx2label,
            average='samples')

        # Generate a multiclass confusion matrix
        labels = set(idx2label.values())
        labels.discard('<PAD>')
        self.cfm = multilabel_confusion_matrix(
            y_true=self.__dict__['gold']['flat']['readable'],
            y_pred=self.__dict__['pred']['flat']['readable'],
            labels=labels
        )

        # Generate a classification report
        self.report = classification_report(
            y_true=self.__dict__['gold']['flat']['readable'],
            y_pred=self.__dict__['pred']['flat']['readable'],
            labels=labels
        )



def create_dict(y_pred: np.ndarray,
                y_gold: np.ndarray,
                X: np.ndarray,
                idx2label: Dict[int, str]) -> Dict[str, Dict[Any, Any]]:

    predictions_dict = {
        'pred': y_dict(X, y_pred, idx2label),
        'gold': y_dict(X, y_gold, idx2label)
    }

    return predictions_dict


def binary_precision(y_true, y_pred):

    true_positives = sum(map(np.logical_and, y_pred, y_true))
    false_posiives = sum(map(np.logical_and, np.logical_not(y_true), y_pred))

    # Deal with zero division issues if they arise
    sum_ = true_positives + false_posiives
    divisor = sum_ if sum_ else sum_ + 10 ** -10
    return true_positives / divisor


def binary_recall(y_true, y_pred) -> float:

    true_positives = sum(map(np.logical_and, y_pred, y_true))
    false_negatives = sum(map(np.logical_and, np.logical_not(y_pred), y_true))

    # Deal with zero division issues if they arise
    sum_ = true_positives + false_negatives
    divisor = sum_ if sum_ else sum_ + 10 ** -10
    return true_positives / divisor


def binary_f1(y_true, y_pred) -> float:
    precision = binary_precision(y_true, y_pred)
    recall = binary_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def binary_scores(y_true, y_pred):

    Binary = namedtuple(
        'Binary', 'precision recall f1'
    )

    precision = binary_precision(y_true, y_pred)
    recall = binary_recall(y_true, y_pred)
    f1 = binary_f1(y_true, y_pred)

    return Binary(precision, recall, f1)


def proportional_scores(y_true, y_pred, idx2label, average: str = 'micro'):

    Proportional = namedtuple(
        'Proportional', 'precision recall f1'
    )

    # Since we don't know which indices correspond to the labels
    # we are interested in, we exclude padding and the majority class,
    # and generate a list containing the rest, in arbitrary order
    excluded = ['<PAD>', 'O']
    labels = [
        idx for (idx, val) in idx2label.items()
        if val not in excluded
    ]

    precision = precision_score(
        y_true, y_pred,
        labels=labels,
        average=average,
        zero_division='warn'  # alternatively 0 or 1
    )
    recall = recall_score(
        y_true, y_pred,
        labels=labels,
        average=average,
        zero_division='warn'  # alternatively 0 or 1
    )
    f1 = f1_score(
        y_true, y_pred,
        labels=labels,
        average=average,
        zero_division='warn'  # alternatively 0 or 1
    )

    return Proportional(precision, recall, f1)


if __name__ == '__main__':
    ones = np.ones((10,), dtype=int)
    zeros = np.zeros((10,), dtype=int)
    more_ones = np.r_[ones[:7], zeros[:3]]
    more_zeros = np.r_[zeros[:7], ones[:3]]

    print(f'ones:       {ones}')
    print(f'zeros:      {zeros}')
    print(f'more_ones:  {more_ones}')
    print(f'more_zeros: {more_zeros}')

    print('\n\n')

    print(f'true: {ones}\npred: {ones}\n{binary_scores(ones, ones)}\n')
    print(f'true: {ones}\npred: {zeros}\n{binary_scores(ones, zeros)}\n')
    print(f'true: {zeros}\npred: {ones}\n{binary_scores(zeros, ones)}\n')
    print(f'true: {zeros}\npred: {zeros}\n{binary_scores(zeros, zeros)}\n')
    print(f'true: {more_ones}\npred: {ones}\n{binary_scores(more_ones, ones)}\n')
    print(f'true: {ones}\npred: {more_ones}\n{binary_scores(ones, more_ones)}\n')
    print(f'true: {more_zeros}\npred: {more_ones}\n{binary_scores(more_zeros, more_ones)}\n')

