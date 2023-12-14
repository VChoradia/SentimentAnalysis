# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""

from helper import read_files, parse_args, compute_confusion_matrix, plot_confusion_matrix
from macro_f1_score import macro_f1_score
from preprocessing import preprocessing
from train_and_predict import train_and_predict


USER_ID = "aca21vvc"

def main():
    inputs = parse_args()

    # input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test

    # number of classes
    number_classes = inputs.classes

    # accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features

    # whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files

    # whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix

    # Reading and preprocessing files
    train, dev, test = read_files(training, dev, test)

    train['Preprocessed_Phrase'] = train['Phrase'].apply(preprocessing)
    dev['Preprocessed_Phrase'] = dev['Phrase'].apply(preprocessing)
    test['Preprocessed_Phrase'] = test['Phrase'].apply(preprocessing)

    if number_classes == 3:
        # Mapping 5-value sentiments to 3-value sentiments
        sentiment_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

        train['Sentiment'] = train['Sentiment'].apply(lambda x: sentiment_mapping[x])
        dev['Sentiment'] = dev['Sentiment'].apply(lambda x: sentiment_mapping[x])

    ALPHA = 1.0

    predict_dev, predict_test = train_and_predict(features, train, dev, test, ALPHA)

    #                         true labels , predicted labels
    f1_score = macro_f1_score(dev["Sentiment"], predict_dev)

    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

    if confusion_matrix:
        matrix = compute_confusion_matrix(dev["Sentiment"], predict_dev, number_classes)
        plot_confusion_matrix(matrix)


if __name__ == "__main__":
    main()
