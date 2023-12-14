import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def combined_features(sentences):
    # sentences = [" ".join(tokens) for tokens in sentences]

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(sentences).toarray()

    # VADER Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    vader_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
    vader_scores = np.array(vader_scores).reshape(-1, 1)

    # Concatenating Features
    # combined_features = torch.cat((X_tfidf.toarray(), vader_scores, bert_embeddings), dim=1)
    combined_features = np.concatenate((X_tfidf, vader_scores), axis=1)

    return combined_features

    # Now use combined_features as input to your classifier (e.g., Multinomial Naive Bayes)
    # Train your classifier and perform evaluation


def combination(train, dev, test):

    train_sentences = train['Preprocessed_Phrase'].tolist()
    dev_sentences = dev['Preprocessed_Phrase'].tolist()
    test_sentences = test['Preprocessed_Phrase'].tolist()

    # Get combined features for each dataset
    train_combined = combined_features(train_sentences)
    dev_combined = combined_features(dev_sentences)
    test_combined = combined_features(test_sentences)

    # Assign combined features back to the respective DataFrame columns
    train['combined_features'] = train_combined.tolist()
    dev['combined_features'] = dev_combined.tolist()
    test['combined_features'] = test_combined.tolist()

    return train, dev, test



