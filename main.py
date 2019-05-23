# download dataset from https://www.kaggle.com/kazanova/sentiment140/downloads/sentiment140.zip/2
import pandas as pd
from pprint import pprint
from random import shuffle
import string
import nltk
from autocorrect import spell
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


class Data:
    def __init__(self, path, n, fraction):
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        df = df.rename(columns={df.columns[0]: 'target',
                                df.columns[1]: 'ids',
                                df.columns[2]: 'date',
                                df.columns[3]: 'flag',
                                df.columns[4]: 'user',
                                df.columns[5]: 'text'})

        df = df[['target', 'text']].sample(n=n)
        self.fraction = fraction
        self.df = df

    def get_data_set(self):
        tuples = [tuple(x) for x in self.df.values]

        data_set_negative = [x for x in tuples if x[0] == 0]
        data_set_afirmative = [x for x in tuples if x[0] == 4]

        shuffle(data_set_afirmative)
        shuffle(data_set_negative)
        train_set = data_set_afirmative[int(self.fraction * len(data_set_afirmative)):] + \
                    data_set_negative[int(self.fraction*len(data_set_negative)):]
        test_set = data_set_afirmative[:int(self.fraction * len(data_set_afirmative))] + \
                   data_set_negative[:int(self.fraction*len(data_set_negative))]

        return test_set, train_set


class NLP:
    @staticmethod
    def tokenize(text):
        if text == '':
            return text

        for c in string.punctuation:
            text = text.replace(c, ' ')

        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token.lower() for token in tokens if token.isalpha()]
        spellchecked_tokens = [spell(x) for x in filtered_tokens]
        return spellchecked_tokens


class NaiveBayesFaultEstimator:
    def __init__(self, data_set):
        self.data = data_set
        self.classifier = self.create_classifier()

    def create_classifier(self):
        self.dict_tag = {k: v for v, k in enumerate(set([d[0] for d in self.data]))}
        self.reverse_dict_tag = {v: k for k, v in self.dict_tag.items()}

        train_x = [' '.join(NLP.tokenize(d[1])) for d in self.data]
        train_y = np.asarray([self.dict_tag[d[0]] for d in self.data], dtype=np.int64)

        classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())
        classifier.fit(train_x, train_y)
        return classifier

    def classify(self, text):
        """Classify text, returns classification name"""
        treated_text = [' '.join(NLP.tokenize(text))]
        prediction_ranking = self.classifier.predict(treated_text)
        return self.reverse_dict_tag[prediction_ranking[0]]


if __name__ == '__main__':
    data = Data(path='dataset/training.1600000.processed.noemoticon.csv', n=1000, fraction=0.2)
    [test, train] = data.get_data_set()
    # pprint('TRAIN SET: ')
    # pprint(train)
    # pprint('TEST SET:')
    # pprint(test)
    estimator = NaiveBayesFaultEstimator(train)

    hits = 0
    for d in test:
        if d[0] == estimator.classify(d[1]):
            hits += 1
    print('Accuracy is:', hits / len(test))
