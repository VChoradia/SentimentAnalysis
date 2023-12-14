"""

Author: Vivek V Choradia

"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocessing(phrase):

    # Lower Case
    phrase = re.sub(r"[A-Z]", lambda x: x.group(0).lower(), phrase)

    # Contractions
    contractions_dict = {
        r"(\b\w+)\s*(n't\b)": "\\1 not",  # Handles "can't", "won't", etc.
        r"(\b\w+)\s*('ll\b)": "\\1 will",  # Handles "I'll", "you'll", etc.
        r"(\b\w+)\s*('ve\b)": "\\1 have",  # Handles "I've", "they've", etc.
        r"(\b\w+)\s*('re\b)": "\\1 are",  # Handles "you're", "we're", etc.
        r"(\b\w+)\s*('d\b)": "\\1 would",  # Handles "he'd", "they'd", etc.
        r"(\bare)\s*(n't\b)": "\\1 not"  # Handles "aren't"
        # Add more patterns as needed
    }

    # for pattern, replacement in contractions_dict.items():
    #     phrase = re.sub(pattern, replacement, phrase)
    #
    # # Tokenising -Adopted from COM3110 Text Processing Lab - Week 8
    pattern = r'''(?x)          # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''

    # phrase = nltk.regexp_tokenize(phrase, pattern)

    # Stop Words
    stop_words = set(stopwords.words('english'))
    # new_phrase = [word for word in phrase if word not in stop_words and word.isalpha()]
    phrase = ' '.join([word for word in phrase.split() if word not in stop_words and word.isalpha()])

    # Stemming
    # stemmer = PorterStemmer()
    # stemmed_words = [stemmer.stem(word) for word in phrase]
    #
    # # Lemmatization
    # lemmatizer = WordNetLemmatizer()
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    return phrase
