from nltk import pos_tag

def extract_adjectives(text):
    tagged = pos_tag(text)
    adjectives = [word for word, pos in tagged if pos in ['JJ', 'JJR', 'JJS']]
    # JJ: Adjective, JJR: Comparative, JJS: Superlative
    return adjectives

def adjectives(train, dev, test):
    # Apply the extract_adjectives function to your datasets
    train['Adjectives'] = train['Phrase'].apply(extract_adjectives)
    dev['Adjectives'] = dev['Phrase'].apply(extract_adjectives)
    test['Adjectives'] = test['Phrase'].apply(extract_adjectives)

    # Flatten the list of adjectives into individual rows
    train_features = train.explode('Adjectives').dropna(subset=['Adjectives'])
    dev_features = dev.explode('Adjectives').dropna(subset=['Adjectives'])
    test_features = test.explode('Adjectives').dropna(subset=['Adjectives'])

    # Create feature sets (X) and labels (y)
    x_train = train_features['Adjectives'].values.reshape(-1, 1)
    x_dev = dev_features['Adjectives'].values.reshape(-1, 1)
    x_test = test_features['Adjectives'].values.reshape(-1, 1)

    return x_train, x_dev, x_test


