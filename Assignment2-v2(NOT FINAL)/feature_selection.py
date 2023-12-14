from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import re

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
