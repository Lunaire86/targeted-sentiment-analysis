#!/usr/bin/env python3
# coding: utf-8
from typing import Dict, List, Any, Union

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.metrics import Metric


class BinaryTruePositives(Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.expand_dims(sample_weight, axis=2)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


class BinaryTrueNegatives(Metric):

    def __init__(self, name='binary_true_negatives', **kwargs):
        super(BinaryTrueNegatives, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.expand_dims(sample_weight, axis=2)
            values = tf.multiply(values, sample_weight)
        self.true_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_negatives


class BinaryFalsePositives(Metric):

    def __init__(self, name='binary_false_positives', **kwargs):
        super(BinaryFalsePositives, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.expand_dims(sample_weight, axis=2)
            values = tf.multiply(values, sample_weight)
        self.false_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_positives


class BinaryFalseNegatives(Metric):

    def __init__(self, name='binary_false_negatives', **kwargs):
        super(BinaryFalseNegatives, self).__init__(name=name, **kwargs)
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.expand_dims(sample_weight, axis=2)
            values = tf.multiply(values, sample_weight)
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.false_negatives



def get_analysis(y_pred, y_gold, sentences) -> Dict[int, Dict[Any, Any]]:
    prediction_analysis = {}

    for sent_id, (prediction, gold, sentence) in enumerate(zip(y_pred, y_gold, sentences)):
        predicted_targets = []
        label = None

        # Targets
        for idx, pred in enumerate(prediction):
            # TODO : is 'pred' a probability?
            if pred > 0:
                if not label:
                    label = [sentence[idx]]
                else:
                    label.append(sentence[idx])
            else:
                if label:
                    predicted_targets.append(label)
                    label = None

        gold_targets = []
        label = None

        # Targets
        for idx, pred in enumerate(gold):
            if pred > 0:
                if not label:
                    label = [sentence[idx]]
                else:
                    label.append(sentence[idx])
            else:
                if label:
                    gold_targets.append(label)
                    label = None

        prediction_analysis[sent_id] = {}
        prediction_analysis[sent_id]["text"] = [w for w in sentence]
        prediction_analysis[sent_id]["true_label"] = {}
        prediction_analysis[sent_id]["true_label"]["target"] = gold_targets
        prediction_analysis[sent_id]["predicted_label"] = {}
        prediction_analysis[sent_id]["predicted_label"]["target"] = predicted_targets

        return prediction_analysis


def proportional_analysis(flat_gold_labels, flat_predictions):
    target_labels = [2, 3, 4, 5]

    print(f'Proportional results:\n{"#" * 80}\n')

    # Targets
    precision = precision_score(
        flat_gold_labels, flat_predictions,
        labels=target_labels,
        average="micro"
    )
    recall = recall_score(
        flat_gold_labels, flat_predictions,
        labels=target_labels,
        average="micro"
    )
    f1 = f1_score(
        flat_gold_labels, flat_predictions,
        labels=target_labels,
        average="micro"
    )

    print(f"Target precision: {precision:.3f}")
    print(f"Target recall: {recall:.3f}")
    print(f"Target F1: {f1:.3f}\n")

    return f1


def binary_precision(flat_binary_pred, flat_binary_gold):
    pred = flat_binary_pred
    gold = flat_binary_gold

    true_positives = sum(map(np.logical_and, pred, gold))
    false_posiives = sum(map(np.logical_and, np.logical_not(gold), pred))

    return true_positives / (true_positives + false_posiives + 10 ** -10)


def binary_recall(flat_binary_pred, flat_binary_gold):
    pred = flat_binary_pred
    gold = flat_binary_gold

    true_positives = sum(map(np.logical_and, pred, gold))
    false_negatives = sum(map(np.logical_and, np.logical_not(pred), gold))

    for _, ann in annotations.items():
        target = ann["target"][annotation_type]
        prediction = ann["prediction"][annotation_type]

        true_positives += binary_true_pos(target, prediction)
        false_negatives += binary_false_neg(target, prediction)

    return true_positives / (true_positives + false_negatives + (10 ** -10))


def binary_f1(anns, anntype="source"):
    prec = binary_precision(anns, anntype)
    rec = binary_recall(anns, anntype)
    return 2 * ((prec * rec) / (prec + rec))

def binary_analysis(prediction_analysis):
    print(f'Binary results:\n{"#" * 80}\n')

    # Targets
    precision = binary_precision(prediction_analysis, "target")
    recall = binary_recall(prediction_analysis, "target")
    f1 = binary_f1(prediction_analysis, "target")

    print(f"Target precision: {precision:.3f}")
    print(f"Target recall: {recall:.3f}")
    print(f"Target F1: {f1:.3f}\n")

    return f1


def evaluate(y_gold, y_pred, sentences):
    """
    Returns the binary and proportional F1 scores of the model on the examples passed via test_loader.
    :param test_loader: torch.utils.data.DataLoader object with
                        batch_size=1
    """
    flat_preds = [int(value) for sublist in y_pred for value in sublist]
    flat_golds = [int(value) for sublist in y_gold for value in sublist]

    analysis = get_analysis(sentences, y_pred, y_gold)
    binary_f1 = binary_analysis(analysis)
    propor_f1 = proportional_analysis(flat_golds, flat_preds)
    return binary_f1, propor_f1


def decode_analysis(prediction_analysis):
    _prediction_analysis = {}
    _prediction_analysis[0] = {}
    _prediction_analysis[0]["text"] = [w for w in ['Lite', 'tight', 'Tempah']]
    _prediction_analysis[0]["true_label"] = {}
    _prediction_analysis[0]["true_label"]["target"] = [1, 1, 5]
    _prediction_analysis[0]["predicted_label"] = {}
    _prediction_analysis[0]["predicted_label"]["target"] = [1, 1, 3]

    # To update with 'source', i.e. textual version of the targets (I think?)
    # Iterate and look up to add entries like in this example:
    _prediction_analysis[0]["true_label"]["source"] = ['O', 'O', 'B-targ-Negative']
    _prediction_analysis[0]["predicted_label"]["source"] = ['O', 'O', 'B-targ-Positive']

    return _prediction_analysis


if __name__ == '__main__':
    # step 0
    y_dev = [[1, 2], [1, 0, 1]]
    predictions = ['?']  # = model.predict()
    sentences = [['this', 'contains'], ['actual', 'sentences']]

    # integer representations
    for sentence in predictions:
        y_pred = [int(np.argmax(_)) for _ in sentence]
    for sentence in y_dev:
        y_gold = [int(np.argmax(_)) for _ in sentence]

    flattened_y_pred = []  # [int(np.argmax(pred)) for pred in flattened]
    flattened_y_gold = []

    # step 1
    # create analysis
    analysis = get_analysis(y_gold, y_pred, sentences)
    analysis[sent_id] = {}
    analysis[sent_id]["text"] = [w for w in sentence]
    analysis[sent_id]["true_label"] = {}
    analysis[sent_id]["true_label"]["target"] = gold_targets
    analysis[sent_id]["predicted_label"] = {}
    analysis[sent_id]["predicted_label"]["target"] = predicted_targets

    # step 2
    # use analysis to calculate binary f1
    binary_f1 = binary_analysis(analysis)

    # step x
    # flatten y_pred and y_gold
    flat_golds = None
    flat_preds = None

    # step y
    # calculate proportional f1 using the flattened lists
    propor_f1 = proportional_analysis(flat_golds, flat_preds)

'''
for sent_id, (prediction, gold, sentence) in enumerate(zip(y_pred['sequential']['vectorised'][:5], y_gold['sequential']['vectorised'][:5], dev.X[:5])):
    # print(f'id: {sent_id}\n'
    #       f'pred: {prediction}\n'
    #       f'gold: {gold}\n'
    #       f'sentence: {sentence}\n')
    predicted_targets = []
    label = None
    # Targets
    for idx, pred in enumerate(prediction):
        if sent_id == 1:
            print(f'idx: {idx}, pred: {pred}')
        if pred > 0:
            if sent_id == 1:
                print(f'pred > 0')
            if not label:
                label = [sentence[idx]]
                if sent_id == 1:
                    print(f'if not label: True\nlabel = [sentence[idx]] -> [sentence[idx]]={[sentence[idx]]}, label={label}')
            else:
                label.append(sentence[idx])
                if sent_id == 1:
                    print(f'else (if not label):\nlabel.append(sentence[idx]) -> label={label}')
        else:
            if sent_id == 1:
                print('pred !> 0 (should mean pred is 0 then)')
            if label:
                predicted_targets.append(label)
                if sent_id == 1:
                    print(f'pred !> 0 and label = True\npredicted_targets.append(label)\nlabel={label}\npredicted_targets={predicted_targets}')
                label = None
                if sent_id == 1:
                    print('label set to None again')

idx: 0, pred: 1
pred > 0
if not label: True
label = [sentence[idx]] -> [sentence[idx]]=['Løvefilm'], label=['Løvefilm']
idx: 1, pred: 1
pred > 0
else (if not label):
label.append(sentence[idx]) -> label=['Løvefilm', 'som']
idx: 2, pred: 1
pred > 0
else (if not label):
label.append(sentence[idx]) -> label=['Løvefilm', 'som', 'ikke']
idx: 3, pred: 1
pred > 0
else (if not label):
label.append(sentence[idx]) -> label=['Løvefilm', 'som', 'ikke', 'biter']
'''