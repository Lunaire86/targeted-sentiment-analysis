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
    y_true: InitVar[np.ndarray]
    y_pred: InitVar[np.ndarray]
    idx2label: InitVar[Dict[int, str]]
    majority_class: InitVar[str] = 'O'

    # Create dictionaries that lets us access predictions
    # and targets in a number of different formats.
    pred: Dict[str, Dict[str, Dict[str, np.ndarray]]] = field(init=False, default_factory=dict)
    gold: Dict[str, Dict[str, Dict[str, np.ndarray]]] = field(init=False, default_factory=dict)

    # Binary and proportional metrics stored as namedtuples
    binary: NamedTuple = field(init=False)
    regular: NamedTuple = field(init=False)
    # samples: NamedTuple = field(init=False)

    # Multiclass confusion matrix
    cfm: np.ndarray = field(init=False)
    cfm_excluding_majority: str = field(init=False)

    # Classification report
    report: str = field(init=False)
    report_excluding_majority: str = field(init=False)

    def __post_init__(self, X, y_true, y_pred, idx2label, majority_class):
        # Use InitVars to create internal dictionary
        self.gold = y_dict(X, y_true, idx2label)
        self.pred = y_dict(X, y_pred, idx2label)

        # Since we don't know which indices correspond to the labels
        # we are interested in, we don't want to hardcode these
        num_labels = [_ for _ in idx2label.keys()]
        str_labels = [_ for _ in idx2label.values()]

        num_classes = [
            idx for (idx, val) in idx2label.items()
            if val != majority_class
        ]
        str_classes = [
            val for (idx, val) in idx2label.items()
            if val != majority_class
        ]

        # Calculate binary scores (minority vs majority)
        self.binary = binary_scores(
            y_true=self.gold['flat']['binary'],
            y_pred=self.pred['flat']['binary']
        )

        # Calculate proportional scores amongst minority classes
        self.regular = proportional_scores(
            y_true=self.gold['flat']['vectorised'],
            y_pred=self.pred['flat']['vectorised'],
            labels=num_classes
        )

        # Generate a multiclass confusion matrix
        labels = [_ for _ in idx2label.values()]
        self.cfm = multilabel_confusion_matrix(
            y_true=self.gold['flat']['readable'],
            y_pred=self.pred['flat']['readable'],
            labels=str_classes
        )

        # Generate a classification report
        self.report = classification_report(
            y_true=self.gold['flat']['readable'],
            y_pred=self.pred['flat']['readable'],
            labels=str_classes
        )

        # Generate a multiclass confusion matrix
        # without including the majority class
        self.cfm_excluding_majority = multilabel_confusion_matrix(
            y_true=self.gold['flat']['readable'],
            y_pred=self.pred['flat']['readable'],
            labels=str_labels
        )

        # Generate a classification report
        # without including the majority class
        self.report_excluding_majority = classification_report(
            y_true=self.gold['flat']['readable'],
            y_pred=self.pred['flat']['readable'],
            labels=str_labels
        )



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

    return Binary(
        round(precision, 4), round(recall, 4), round(f1, 4)
    )


def proportional_scores(y_true, y_pred, labels):

    Proportional = namedtuple(
        'Proportional', 'precision recall f1'
    )

    precision = precision_score(
        y_true, y_pred,
        labels=labels,
        average='micro'
    )
    recall = recall_score(
        y_true, y_pred,
        labels=labels,
        average='micro'
    )
    f1 = f1_score(
        y_true, y_pred,
        labels=labels,
        average='micro'
    )

    return Proportional(
        round(precision, 4), round(recall, 4), round(f1, 4)
    )


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

    idx2lab = {
        1: 'O',
        2: 'B',
        3: 'A',
        4: 'C'
    }

    true_labels = [
        ['O', 'O', 'O', 'O', 'O'],
        ['A', 'O', 'O', 'O', 'A'],
        ['O', 'B', 'O', 'O', 'C'],
        ['O', 'O', 'O', 'O', 'C']
    ]

    gold = [
        [1, 1, 1, 1, 1],
        [3, 1, 1, 1, 3],
        [1, 2, 1, 1, 4],
        [1, 1, 1, 1, 4]
    ]

    pred = [
        [1, 1, 1, 4, 1],
        [2, 1, 1, 1, 3],
        [1, 2, 1, 1, 4],
        [1, 3, 3, 2, 4]
    ]

    flat_pred = np.array([
        _ for tokens
        in pred
        for _ in tokens
    ])

    flat_gold = np.array([
        _ for tokens
        in gold
        for _ in tokens
    ])


