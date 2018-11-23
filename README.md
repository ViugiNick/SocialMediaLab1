# SocialMediaLab1

Console application for tweet classification:

## Modes
1. sentiment - Sentiment classification using tweets
2. company - Company classification using tweets
3. mixed - Sentiment classification using tweets and companies

## Modes
You can specify train and test files (--train, --test options) and mode(--mode option)

After running the application, just select classifier and follow the instructions

type 'exit' to stop application

## Usage example
    !python3 main.py --mode=sentiment
    Select classifier:
    1 - SentiWordNet dictionary classifier
    2 - RandomForest classifier
    3 - Bayes classifier
    3
    You selected Bayes classifier
    Classifier accuracy:
    0.7510548523206751
                 precision    recall  f1-score   support

    negative       0.75      0.49      0.59        49
    neutral        0.77      0.93      0.84       156
    positive       0.56      0.28      0.38        32

    avg / total    0.74      0.75      0.73       237

    [[ 24  22   3]
     [  7 145   4]
     [  1  22   9]]

    Enter your tweet:
    What a nice example
    positive
