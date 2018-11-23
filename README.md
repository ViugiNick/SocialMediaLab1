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
    Enter your tweet:
    What a nice example
    positive
