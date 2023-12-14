# Function to handle negation
import string

import pandas as pd


def handle_negation(text: pd.Series):

    negation_markers = ["not", "no", "never", "hard", "bare", "scarce", "hardly ever", "seldom"]  # Add more if needed
    punctuation = set(string.punctuation)
    result = []
    negated = False
    for word in text:
        if negated and word in punctuation:
            negated = False

        if negated:
            result.append('NOT_' + word)
        else:
            result.append(word)

        if word.lower() in negation_markers:
            negated = True

    return result


def negation(train, dev, test):
    # Apply the extract_adjectives function to your datasets
    train['NegHandled'] = train['Phrase'].apply(handle_negation)
    dev['NegHandled'] = dev['Phrase'].apply(handle_negation)
    test['NegHandled'] = test['Phrase'].apply(handle_negation)

    # Flatten the list of adjectives into individual rows
    train_features = train.explode('NegHandled').dropna(subset=['NegHandled'])
    dev_features = dev.explode('NegHandled').dropna(subset=['NegHandled'])
    test_features = test.explode('NegHandled').dropna(subset=['NegHandled'])

    # Create feature sets (X) and labels (y)
    x_train = train_features['NegHandled'].values.reshape(-1, 1)
    x_dev = dev_features['NegHandled'].values.reshape(-1, 1)
    x_test = test_features['NegHandled'].values.reshape(-1, 1)

    return x_train, x_dev, x_test
