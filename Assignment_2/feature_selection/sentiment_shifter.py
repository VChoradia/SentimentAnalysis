# Function to handle sentiment shifters
def handle_sentiment_shifters(text):
    shifter_markers = {"but", "however", "although", "though", "yet", "while", "despite", "nevertheless", "nonetheless"}

    shifted = []
    for i, word in enumerate(text):
        if word in shifter_markers:
            shifted.extend(text[i:])
            break
        shifted.append(word)
    return shifted


def sentiment_shifters(train, dev, test):

    # Apply the extract_adjectives function to your datasets
    train['Sentiment'] = train['Phrase'].apply(handle_sentiment_shifters)
    dev['Sentiment'] = dev['Phrase'].apply(handle_sentiment_shifters)
    test['Sentiment'] = test['Phrase'].apply(handle_sentiment_shifters)

    # Flatten the list of adjectives into individual rows
    train_features = train.explode('Sentiment').dropna(subset=['Sentiment'])
    dev_features = dev.explode('Sentiment').dropna(subset=['Sentiment'])
    test_features = test.explode('Sentiment').dropna(subset=['Sentiment'])

    # Create feature sets (X) and labels (y)
    x_train = train_features['Sentiment'].values.reshape(-1, 1)
    x_dev = dev_features['Sentiment'].values.reshape(-1, 1)
    x_test = test_features['Sentiment'].values.reshape(-1, 1)

    return x_train, x_dev, x_test
