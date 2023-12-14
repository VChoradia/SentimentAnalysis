import nltk
import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def vader_sentiment_scores(text):

    sia = SentimentIntensityAnalyzer()
    vader_scores = [sia.polarity_scores(sentence)['compound'] for sentence in text]
    vader_scores = np.array(vader_scores).reshape(-1, 1)

    return vader_scores

def vader_sentiment(train, dev, test):

    train_sentences = train['Phrase'].tolist()
    dev_sentences = dev['Phrase'].tolist()
    test_sentences = test['Phrase'].tolist()

    x_train = vader_sentiment_scores(train_sentences)
    x_dev = vader_sentiment_scores(dev_sentences)
    print(x_dev)
    x_test = vader_sentiment_scores(test_sentences)

    return x_train, x_dev, x_test
