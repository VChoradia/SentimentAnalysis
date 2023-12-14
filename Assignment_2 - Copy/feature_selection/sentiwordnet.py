import pandas as pd
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return NOUN  # Default to noun

def get_sentiwordnet_scores(text):
    tagged_tokens = pos_tag(text)
    pos_score = 0
    neg_score = 0
    token_count = 0

    for token, treebank_tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(treebank_tag)
        synsets = wn.synsets(token, pos=wordnet_pos)
        if synsets:
            synset = synsets[0]
            senti_synset = swn.senti_synset(synset.name())
            pos_score += senti_synset.pos_score()
            neg_score += senti_synset.neg_score()
            token_count += 1

    if token_count > 0:
        pos_score /= token_count
        neg_score /= token_count

    sentiment = pos_score - neg_score

    return sentiment

def sentiwordnet(number_classes, train, dev, test) -> (pd.Series, pd.Series, pd.Series):

    # Assuming 'Phrase' column contains text data
    # x_train = train['Preprocessed_Phrase'].apply(get_sentiwordnet_scores)
    # y_train = train[number_classes]
    # x_dev = dev['Preprocessed_Phrase'].apply(get_sentiwordnet_scores)
    # x_test = test['Preprocessed_Phrase'].apply(get_sentiwordnet_scores)
    #
    # return x_train, y_train, x_dev, x_test

    train_sentences = train['Preprocessed_Phrase']
    dev_sentences = dev['Preprocessed_Phrase']
    test_sentences = test['Preprocessed_Phrase']

    print(train_sentences)

    x_train = get_sentiwordnet_scores(train_sentences)
    x_dev = get_sentiwordnet_scores(dev_sentences)
    x_test = get_sentiwordnet_scores(test_sentences)

    y_train = train[number_classes]

    return x_train, y_train, x_dev, x_test
