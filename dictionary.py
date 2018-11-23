import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

class DictionarySentiment(object):
    def __init__(self, trainData, l, r):
        self.tweets, self.labels = trainData['TweetText'], trainData['Sentiment']
        self.data = trainData
        self.l = l
        self.r = r
        self.wnl = nltk.WordNetLemmatizer()
        self.sw = stopwords.words('english')

    def classify(self, message):
        message = message.strip()
        #message = ''.join([char for char in message if char not in '@#"\''])
        message = " ".join([word for word in message.split()
                            if 'http' not in word
                            ])

        sentences = nltk.sent_tokenize(message)
        stokens = [nltk.word_tokenize(sent) for sent in sentences]
        stokens = [sent for sent in stokens if sent not in self.sw]

        taggedlist = []
        for stoken in stokens:
            taggedlist.append(nltk.pos_tag(stoken))

        score_list = []
        for idx, taggedsent in enumerate(taggedlist):
            score_list.append([])
            for idx2, t in enumerate(taggedsent):
                lemmatized = self.wnl.lemmatize(t[0])
                if t[1].startswith('NN'):
                    newtag = 'n'
                elif t[1].startswith('JJ'):
                    newtag = 'a'
                elif t[1].startswith('V'):
                    newtag = 'v'
                elif t[1].startswith('R'):
                    newtag = 'r'
                else:
                    newtag = ''
                if (newtag != ''):
                    synsets = list(swn.senti_synsets(lemmatized, newtag))
                    score = 0
                    if (len(synsets) > 0):
                        for syn in synsets:
                            score += syn.pos_score() - syn.neg_score()
                        score_list[idx].append(score / len(synsets))

        ans = 0

        for score_sent in score_list:
            if (len(score_sent) != 0):
                ans += sum([word_score for word_score in score_sent]) / len(score_sent)

        if ans > self.r:
            return 'positive'
        if ans < self.l:
            return 'negative'
        return 'neutral'

    def predict(self, testData):
        result = []

        for i, row in testData.iterrows():
            result.append(self.classify(row['TweetText']))

        return result