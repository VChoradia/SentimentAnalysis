import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def vader_sentiment_scores(text):
    # Analyze sentiment and return expanded feature set including compound, neg, neu, and pos scores
    features = []
    for sentence in text:
        scores = sia.polarity_scores(sentence)
        # Append all scores for a more nuanced feature set
        features.append([scores['compound'], scores['neg'], scores['neu'], scores['pos']])
    return np.array(features)

def extract_tfidf_features(sentences):
    # Use TF-IDF to vectorize the sentences, consider using bigrams as well
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_tfidf = vectorizer.fit_transform(sentences)
    return X_tfidf

def reduce_dimensions(X):
    # Reduce the dimensionality of TF-IDF features
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_X = svd.fit_transform(X)
    return reduced_X

def combined_sentiment(number_classes, train, dev, test):
    # Extract sentences
    train_sentences = train['Preprocessed_Phrase'].tolist()
    dev_sentences = dev['Preprocessed_Phrase'].tolist()
    test_sentences = test['Preprocessed_Phrase'].tolist()

    # Generate VADER sentiment scores
    x_train_vader = vader_sentiment_scores(train_sentences)
    x_dev_vader = vader_sentiment_scores(dev_sentences)
    x_test_vader = vader_sentiment_scores(test_sentences)

    # Generate TF-IDF features
    x_train_tfidf = extract_tfidf_features(train_sentences)
    x_dev_tfidf = extract_tfidf_features(dev_sentences)
    x_test_tfidf = extract_tfidf_features(test_sentences)

    # Dimensionality reduction
    x_train_reduced = reduce_dimensions(x_train_tfidf)
    x_dev_reduced = reduce_dimensions(x_dev_tfidf)
    x_test_reduced = reduce_dimensions(x_test_tfidf)

    # Combine VADER and TF-IDF features
    x_train_combined = np.hstack((x_train_vader, x_train_reduced))
    x_dev_combined = np.hstack((x_dev_vader, x_dev_reduced))
    x_test_combined = np.hstack((x_test_vader, x_test_reduced))

    # Get the labels for training data
    y_train = train[number_classes]

    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_combined, y_train)

    return x_train_balanced, y_train_balanced, x_dev_combined, x_test_combined

