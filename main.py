from optparse import OptionParser
import pandas as pd
from dictionary import DictionarySentiment
from bayes import MyBayesClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

def getDictionarySentiment(train_data, test_data):
    dictSentiment = DictionarySentiment(test_data, -0.1, 0.485)
    dictAns = dictSentiment.predict(test_data)

    print('SentiWordNet accuracy:')
    print(accuracy_score(test_data['Sentiment'], dictAns))

    return dictSentiment

def getClassifier(train_data, test_data, text_name = 'TweetText', label_name = 'Sentiment', clf = None):
    bayesClf = MyBayesClassifier(text_name, label_name, clf)

    bayesClf.fit(train_data)
    bayesAns = bayesClf.predict(test_data)

    print('Classifier accuracy:')
    print(accuracy_score(test_data[label_name], bayesAns))
    print(confusion_matrix(test_data['Sentiment'], bayesAns))

    return bayesClf

parser = OptionParser()
parser.add_option("--train", dest="train_file", default="Train.csv",
                  help="train .scv filepath", metavar="FILE")
parser.add_option("--test", dest="test_file", default="Test.csv",
                  help="test .scv filepath", metavar="FILE")
parser.add_option("-m", "--mode", dest="mode", default="sentiment",
                  help="usage mode, can be: sentiment, company, mixed")

(options, args) = parser.parse_args()

train_data = pd.read_csv(options.train_file, encoding='utf-8').sample(frac=1).reset_index(drop=True)
train_data = train_data[train_data['Sentiment'] != 'irrelevant']

test_data = pd.read_csv(options.test_file, encoding='utf-8').sample(frac=1).reset_index(drop=True)
test_data = test_data[test_data['Sentiment'] != 'irrelevant']

if options.mode == 'sentiment':
    print('Select classifier:')
    print('1 - SentiWordNet dictionary classifier')
    print('2 - RandomForest classifier')
    print('3 - Bayes classifier')

    clf_id = int(input())
    assert (clf_id == 1 or clf_id == 2 or clf_id ==3)

    if clf_id == 1:
        print('You selected SentiWordNet dictionary classifier')
        clf = getDictionarySentiment(train_data, test_data)
    if clf_id == 2:
        print('You selected RandomForest classifier')
        clf = getClassifier(train_data, test_data, clf = RandomForestClassifier(n_estimators = 100))
    if clf_id == 3:
        print('You selected Bayes classifier')
        clf = getClassifier(train_data, test_data, clf = MultinomialNB(alpha=0.1, fit_prior=True))

    while True:
        print('Enter your tweet:')
        tweet = input()
        if tweet == 'exit':
            break
        print(clf.classify(tweet))

if options.mode == 'company':
    print('Select classifier:')
    print('1 - RandomForest classifier')
    print('2 - Bayes classifier')

    clf_id = int(input())
    assert (clf_id == 1 or clf_id == 2)

    if clf_id == 1:
        print('You selected RandomForest classifier')
        clf = getClassifier(train_data, test_data, text_name = 'TweetText', label_name = 'Topic', clf = RandomForestClassifier())
    if clf_id == 2:
        print('You selected Bayes classifier')
        clf = getClassifier(train_data, test_data, text_name = 'TweetText', label_name = 'Topic', clf = MultinomialNB(alpha=0.1, fit_prior=True))

    while True:
        print('Enter your tweet:')
        tweet = input()
        if tweet == 'exit':
            break
        print(clf.classify(tweet))

if options.mode == 'mixed':
    print('Select classifier:')
    print('1 - RandomForest classifier')
    print('2 - Bayes classifier')

    clf_id = int(input())
    assert (clf_id == 1 or clf_id == 2)

    if clf_id == 1:
        print('You selected RandomForest classifier')
        clf = getClassifier(train_data, test_data, text_name = 'TweetText', label_name = 'Topic', clf = RandomForestClassifier())
    if clf_id == 2:
        print('You selected Bayes classifier')
        clf = getClassifier(train_data, test_data, text_name = 'TweetText', label_name = 'Topic', clf = MultinomialNB(alpha=1, fit_prior=True))

    while True:
        print('Enter your tweet:')
        tweet = input()
        if tweet == 'exit':
            break
        print(clf.classify(tweet))