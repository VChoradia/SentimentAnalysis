# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse

from helper import read_files
# from macro_f1_score import macro_f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from macro_f1_score import macro_f1_score

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca21vvc" # your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE

def parse_args():
    parser = argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser "
                                               "for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


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

    train, dev, test = read_files(training, dev, test)

    # Preprocessing
    # Feature Selection - negation, n-gram, mutual information

    if number_classes == 3:
        # Mapping 5-value sentiments to 3-value sentiments
        sentiment_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

        train['Sentiment'] = train['Sentiment'].apply(lambda x: sentiment_mapping[x])
        dev['Sentiment'] = dev['Sentiment'].apply(lambda x: sentiment_mapping[x])

    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        # Convert text to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenization
        words = nltk.word_tokenize(text)

        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]

        # Lemmatization
        words = [lemmatizer.lemmatize(word) for word in words]

        # Re-create document from words
        text = " ".join(words)
        return text

    # processed_texts = [preprocess_text(text) for text in sample_texts]
    # tfidf_vectors = vectorize_texts(processed_texts)

    train['Phrase'] = train['Phrase'].apply(preprocess_text)
    dev['Phrase'] = dev['Phrase'].apply(preprocess_text)

    # Negation handling
    def handle_negation(documents):
        transformed_docs = []
        negation_words = set(["not", "n't", "no", "never", "cannot"])
        neg_endings = set(['.', ',', '!', '?'])

        for doc in documents:
            words = doc.split()
            transformed_words = []
            negation = False
            for word in words:
                if negation:
                    transformed_words.append('NEG_' + word)
                    if word in neg_endings:
                        negation = False
                else:
                    if word in negation_words:
                        negation = True
                    transformed_words.append(word)
            transformed_docs.append(' '.join(transformed_words))
        return transformed_docs

    # N-Gram extraction
    def extract_ngrams(data, n=2):
        count_vect = CountVectorizer(ngram_range=(1, n))
        return count_vect.fit_transform(data)

    # Mutual Information
    def select_features_by_mutual_info(X, y, k=500):
        selector = SelectKBest(mutual_info_classif, k=k)
        return selector.fit_transform(X, y)

    # Preprocessing and feature selection
    negated_train = handle_negation(train["Phrase"])
    X_train = extract_ngrams(negated_train, n=2)  # Using bigrams
    X_train_selected = select_features_by_mutual_info(X_train, train["Sentiment"], k=1000)

    # Multinomial Naive Bayes Classifier with Laplace Smoothing
    nb_classifier = MultinomialNB(alpha=1.0)  # alpha=1.0 for Laplace smoothing

    # Fit the model
    nb_classifier.fit(X_train_selected, train["Sentiment"])

    # Predictions and Evaluation
    negated_dev = handle_negation(dev["Phrase"])
    X_dev = extract_ngrams(negated_dev, n=2)
    X_dev_selected = select_features_by_mutual_info(X_dev, dev["Sentiment"], k=1000)

    dev_predictions = nb_classifier.predict(X_dev_selected)
    print(dev_predictions)
    # f1_score_dev = f1_score()

    # print("Macro F1 Score on Dev Set:", f1_score_dev)

    # You need to change this in order to return your macro-F1 score for the dev set
    f1_score = macro_f1_score(dev["Sentiment"], dev_predictions)

    # print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()
