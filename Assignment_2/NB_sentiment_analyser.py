# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""

from helper import read_files, parse_args, compute_confusion_matrix, plot_confusion_matrix, create_prediction_files
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

    train['Phrase'] = train['Phrase'].apply(preprocessing)
    dev['Phrase'] = dev['Phrase'].apply(preprocessing)
    test['Phrase'] = test['Phrase'].apply(preprocessing)

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

    if output_files:
        create_prediction_files(dev['SentenceId'], predict_dev, number_classes, 'dev', USER_ID)
        create_prediction_files(test['SentenceId'], predict_test, number_classes, 'test', USER_ID)

    # Error Analysis
    # errors = dev[dev["Sentiment"] != predict_dev]
    # errors['Predicted'] = predict_dev[dev["Sentiment"] != predict_dev]
    # errors['SentenceId'] = dev['SentenceId']
    # print("Error Analysis: True vs Predicted Sentiments")
    # print(errors[['Sentiment', 'Predicted', 'SentenceId']])

    specific_error_condition = (dev["Sentiment"] == 1) & (predict_dev == 4)
    specific_errors = dev[specific_error_condition]
    specific_errors['Predicted'] = predict_dev[specific_error_condition]
    print("Specific Error Analysis: SentenceId, True (1) vs Predicted (4) Sentiments")
    print(specific_errors[['SentenceId', 'Sentiment', 'Predicted']])


if __name__ == "__main__":
    main()
