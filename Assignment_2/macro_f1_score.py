"""
Author: Vivek V Choradia
"""
import pandas as pd


def calculate_f1_score(true_labels: pd.Series, predicted_labels: pd.Series, label) -> float:
    """
    Calculate F1 Score for each label
    :param true_labels:
    :param predicted_labels:
    :param label:
    :return: f1 score for each label
    """

    true_positives = ((true_labels == label) & (predicted_labels == label)).sum()
    false_positives = ((true_labels != label) & (predicted_labels == label)).sum()
    false_negatives = ((true_labels == label) & (predicted_labels != label)).sum()

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1


def macro_f1_score(true_labels: pd.Series, predicted_labels: pd.Series) -> [float]:
    """

    :param true_labels:
    :param predicted_labels:
    :return: Macro f1 score for the feature
    """

    unique_labels = true_labels.unique()

    f1_scores = [calculate_f1_score(true_labels, predicted_labels, label) for label in unique_labels]

    macro_f1 = pd.Series(f1_scores).mean()

    return macro_f1






