#!/usr/bin/env python3
# coding: utf-8

from sklearn.metrics import precision_score, recall_score, f1_score


def binary_true_pos(target, prediction):
    """
    For each member in prediction, if it overlaps with any member of target,
    return 1
    else return 0
    """
    true_positives = 0

    for pred in prediction:
        true_positive = False

        for word in pred:
            for span in target:

                if word in span:
                    true_positive = True

        if true_positive is True:
            true_positives += 1

    return true_positives


def binary_false_neg(target, prediction):
    """
    If there is any member of target that overlaps with no member of prediction,
    return 1
    else return 0
    """
    false_negatives = 0

    for pred in target:
        false_negative = True

        for word in pred:
            for span in prediction:

                if word in span:
                    false_negative = False

        if false_negative is True:
            false_negatives += 1

    return false_negatives


def binary_false_pos(target, prediction):
    """
    If there is any member of prediction that overlaps with
    no member of target,
    return 1
    else return 0
    """
    false_positves = 0

    for pred in prediction:
        false_positive = True

        for word in pred:
            for span in target:

                if word in span:
                    false_positive = False

        if false_positive is True:
            false_positves += 1

    return false_positves


def binary_precision(annotations, annotation_type='source'):
    true_positives = 0
    false_negatives = 0

    for _, ann in annotations.items():
        target = ann["target"][annotation_type]
        prediction = ann["prediction"][annotation_type]

        true_positives += binary_true_pos(target, prediction)
        false_negatives += binary_false_pos(target, prediction)

    return true_positives / (true_positives + false_negatives + 10 ** -10)


def binary_recall(annotations, annotation_type='source'):
    true_positives = 0
    false_negatives = 0

    for _, ann in annotations.items():
        target = ann["target"][annotation_type]
        prediction = ann["prediction"][annotation_type]

        true_positives += binary_true_pos(target, prediction)
        false_negatives += binary_false_neg(target, prediction)

    return true_positives / (true_positives + false_negatives + 10 ** -10)


def binary_f1(annotations, annotation_type='source'):
    precision = binary_precision(annotations, annotation_type)
    recall = binary_recall(annotations, annotation_type)

    return 2 * ((precision * recall) / (precision + recall))


def binary_analysis(prediction_analysis):
    print("Binary results:")
    print("#" * 80)
    print()

    # Targets
    precision = binary_precision(prediction_analysis, "target")
    recall = binary_recall(prediction_analysis, "target")
    f1 = binary_f1(prediction_analysis, "target")

    print(f"Target precision: {precision:.3f}")
    print(f"Target recall: {recall:.3f}")
    print(f"Target F1: {f1:.3f}\n")

    return f1


def proportional_analysis(flat_gold_labels, flat_predictionictions):
    target_labels = [1, 2, 3, 4]

    print("Proportional results:")
    print("#" * 80)
    print()

    # Targets
    precision = precision_score(flat_gold_labels, flat_predictionictions,
                                labels=target_labels, average="micro")
    recall = recall_score(flat_gold_labels, flat_predictionictions,
                          labels=target_labels, average="micro")
    f1 = f1_score(flat_gold_labels, flat_predictionictions,
                  labels=target_labels, average="micro")

    print(f"Target precision: {precision:.3f}")
    print(f"Target recall: {recall:.3f}")
    print(f"Target F1: {f1:.3f}\n")

    return f1


def get_analysis(sentences, y_prediction, y_test):
    prediction_analysis = {}

    for idx, (sentence, prediction, target) in enumerate(zip(sentences, y_prediction, y_test)):
        target = []
        label = None

        # Targets
        for idx, pred in enumerate(prediction):
            if pred > 0:
                if not label:
                    label = [sentence[idx]]
                else:
                    label.append(sentence[idx])
            else:
                if label:
                    target.append(label)
                    label = None

        gold_target = []
        label = None

        # Targets
        for idx, pred in enumerate(target):
            if pred > 0:
                if not label:
                    label = [sentence[idx]]
                else:
                    label.append(sentence[idx])
            else:
                if label:
                    gold_target.append(label)
                    label = None

        prediction_analysis[idx] = {}
        prediction_analysis[idx]["sentence"] = [token for token in sentence]
        prediction_analysis[idx]["gold"] = {}
        prediction_analysis[idx]["gold"]["target"] = gold_target
        prediction_analysis[idx]["prediction"] = {}
        prediction_analysis[idx]["prediction"]["target"] = target

    return prediction_analysis