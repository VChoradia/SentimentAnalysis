import re

def handle_negation(text):
    negation_words = ["not", "n't"]
    punctuation_marks = ['.', '?', '!', ',']
    transformed_words = []
    negated = False

    words = text.split()
    for word in words:
        # Check for punctuation marks to reset negation
        if any(mark in word for mark in punctuation_marks):
            negated = False

        # Check for negation words
        if any(neg_word in word for neg_word in negation_words):
            negated = not negated
            continue

        # Transform word if negated
        if negated:
            word = "not_" + word

        transformed_words.append(word)

    return ' '.join(transformed_words)

def negation(train, dev, test):
    train['Phrase'] = train['Phrase'].apply(handle_negation)
    dev['Phrase'] = dev['Phrase'].apply(handle_negation)
    test['Phrase'] = test['Phrase'].apply(handle_negation)
