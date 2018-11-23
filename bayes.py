import numpy as np
from nltk.corpus import stopwords
from math import log
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class MyBayesClassifier(object):
    def __init__(self, text_name, label_name, clf):
        self.text_name = text_name
        self.label_name = label_name

        self.NBclassifier = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', clf),
                                 ])

    def fit(self, train):
        train_x = train[self.text_name].values.tolist()
        messages = []
        for tweet in train_x:
            tweet = " ".join(self.process_tweet(tweet))
            messages.append(tweet)

        train_y = train[self.label_name].values.tolist()

        self.NBclassifier.fit(messages, train_y)


    def process_tweet(self, message, stem=True, stop_words=True, gram = 0, stemmer=PorterStemmer()):
        message = ''.join([char for char in message if char not in '@#'])

        message = " ".join([word for word in message.split()
                                 if 'http' not in word
                                 ])

        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(message)

        if stop_words:
            sw = stopwords.words('english')
            words = [word for word in words if word not in sw]

        if stem:
            words = [stemmer.stem(word) for word in words]

        if gram > 1:
            w = []
            for i in range(len(words) - gram + 1):
                w += [' '.join(words[i:i + gram])]
            return w

        return words

    def predict(self, test):
        test_x = test[self.text_name].values.tolist()
        test_x = [" ".join(self.process_tweet(word)) for word in test_x]

        return self.NBclassifier.predict(test_x)

    def classify(self, message):
        return self.NBclassifier.predict([message])[0]
