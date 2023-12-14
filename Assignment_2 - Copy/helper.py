import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    return args


def read_file(file_path):
    df = pd.read_csv(file_path, sep="\t")
    return df

def read_files(train_file, dev_file, test_file):
    train = read_file(train_file)
    dev = read_file(dev_file)
    test = read_file(test_file)
    return train, dev, test

def compute_confusion_matrix(true, pred, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        matrix[t][p] += 1
    return matrix

def plot_confusion_matrix(matrix):
    sns.heatmap(matrix, annot=True, fmt='g', cmap='coolwarm')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

