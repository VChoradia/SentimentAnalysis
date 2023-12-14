import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

from feature_selection.combined2 import combined_sentiment

nltk.download('punkt')

from MultinomialNaiveBayes import MultinomialNaiveBayes
from feature_selection.adjectives import adjectives
from feature_selection.combined import combination
from feature_selection.handle_negation import negation
from feature_selection.sentiment_shifter import sentiment_shifters
from feature_selection.sentiwordnet import sentiwordnet
from feature_selection.vader import vader_sentiment


def train_and_predict(feature: str, train: pd.DataFrame,
                      dev: pd.DataFrame, test: pd.DataFrame, alpha: float) -> (pd.Series, pd.Series):

    if feature == "all_words":
        # Extracting features and labels
        x_train = train['Preprocessed_Phrase']
        y_train = train['Sentiment']
        x_dev = dev['Preprocessed_Phrase']
        x_test = test['Preprocessed_Phrase']

    else:
        # Adjectives - reduces the score & doesn't take sentiment relations into account
        # x_train, y_train, x_dev, x_test = adjectives(train, dev, test)

        # negation marker - explicit negation markers not helpful
        # x_train, y_train, x_dev, x_test = negation(train, dev, test)

        # sentiment shifters
        # x_train, y_train, x_dev, x_test = sentiment_shifters(train, dev, test)

        # combining - adjectives, negation marker and sentiment shifters? no need already covered by VADER

        # VADER (Valence Aware Dictionary and sEntiment Reasoner)
        x_train, x_dev, x_test = vader_sentiment(train, dev, test)
        y_train = train['Sentiment']

        # Combined:TFIDF, VADER - reduces the score because TFIDF dilutes sentiment specific information
        # train, dev, test = combination(train, dev, test)
        # x_train = train['combined_features']
        # y_train = train[number_classes]
        # x_dev = dev['combined_features']
        # x_test = test['combined_features']

        # Combined TFIDF, VADER with oversampling using SMOTE - takes longer than expected
        # x_train, y_train, x_dev, x_test = combined_sentiment('Sentiment', train, dev, test)

    # MultinomialNaiveBayes classifier
    naive_bayes_classifier = MultinomialNaiveBayes(alpha=alpha)
    naive_bayes_classifier.fit(x_train, y_train)
    dev_predictions = naive_bayes_classifier.predict(x_dev)
    test_predictions = naive_bayes_classifier.predict(x_test)

    return dev_predictions, test_predictions
