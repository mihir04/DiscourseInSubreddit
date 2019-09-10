"""
@author: Mihir Naresh Shah, Swapnil Sachin Shah
Description: Classification of 2 or more subreddits using SVM and Random Forest classifier
            for multiclass classification. Using nltk to preprocess, clean, remove stopwords, stem,
            tokenize the data before classifying.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from autocorrect import spell
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt



def Cleaning(dataset):
    """
    :param dataset:
    :return:
    """
    nltk.download('stopwords')
    #Taking titles of the subreddits as the attribute for classifications
    titles = dataset.loc[:,'titles']
    corpus = []
    stemmer = PorterStemmer()
    #We clean the data for each and every word in a title or the body
    for word in titles:
        #Remove the punctuations from the data
        subreddit = re.sub('[^A-Za-z]',' ',word)

        #Convert all the data into lowercase for consistency
        subreddit = subreddit.lower()

        #Tokenize each and every word in the subreddit
        tokenized_subreddit = word_tokenize(subreddit)

        #Remove the stopwords from the subreddit
        for x in tokenized_subreddit:
            if x in stopwords.words('english'):
                tokenized_subreddit.remove(x)

        #Stemming the data
        for i in range(len(tokenized_subreddit)):
            tokenized_subreddit[i] = stemmer.stem(tokenized_subreddit[i])
            tokenized_subreddit[i] = stemmer.stem(spell(tokenized_subreddit[i]))

        #Finally each and every word of the sentence is joined and the whole sentence is
        #stored in a list as a strind.
        tokenized_subreddit = ' '.join(tokenized_subreddit)

        #Contains a list of preproccessed sentences.
        corpus.append(tokenized_subreddit)
    SVMclassifier(corpus, dataset)
    RFclassifier(corpus, dataset)


def SVMclassifier(corpus, dataset):
    """
    :param corpus:
    :param dataset:
    :return:
    SVM classifier model
    """
    #Selects the maximum features for classificationc
    cv = CountVectorizer(max_features=100)
    X = cv.fit_transform(corpus).toarray() #The cleaned data is fit and transformed for our model

    #Data is split into training and testing set.as 50%(training data) + 25%(Development data) = 75%
    X_train, X_test, y_train, y_test = train_test_split(X, dataset.loc[:, 'DataSet'], test_size= 0.25, random_state= 0)


    from sklearn.svm import SVC
    #One vs All classifier is used since we have multiple classes
    svc = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    svc.fit(X_train, y_train) #fitting of training data

    #Procedure for drawing the learning curves.
    #The following snippet for plotting the learning curve was referred from the actual documentation
    #and citations 8) and 9) from the report.
    training_size = [200, 500, 900, 1000]
    learning_curve_SVM(svc, X_train, X_test, y_train, y_test, training_size)
    Y_pred = svc.predict(X_test)

    # Calculating the accuracy and confusion matrix.
    from sklearn.metrics import accuracy_score
    print("SVM Classifier:")
    print("Accuracy percentage: ", accuracy_score(y_test, Y_pred) * 100, "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, Y_pred))
    PrecisionAndRecallSVM(y_test, Y_pred)


def learning_curve_SVM(classifier, X_train, X_test, y_train, y_test, training_size):
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=5, scoring='accuracy',
                                                            n_jobs=-1, train_sizes=training_size)
    train_mean = np.mean(train_scores, axis=1)
    x1 = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    x2 = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="red", label="Training values")
    plt.plot(train_sizes, test_mean, '--', color="blue", label="Cross-validation values")

    plt.title("Learning Curve for SVM")
    plt.xlabel("Size of training set")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()



#Random forest
def RFclassifier(corpus, dataset):
    """
    :param corpus:
    :param dataset:
    :return:
    Random Forest Classifier
    """
    # Selects the maximum features for classificationc
    cv = CountVectorizer(max_features=100)

    #X and Y both are needed to be fit and transformed for Random Forest
    X = cv.fit_transform(corpus).toarray()
    Y = cv.fit_transform(dataset.loc[:, 'DataSet']).toarray()

    #Split the data into training and testing sets as 50%(training data) + 25%(Development data) = 75% data for
    #training and  25% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, random_state= 0)


    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)#fit the training data

    #The following snippet for plotting the learning curve was referred from the actual documentation
    #and citations 8) and 9) from the report.
    training_size = [200, 500, 900, 100]
    learning_curve_RF(classifier, X_train, X_test, y_train, y_test, training_size)

    Y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score, confusion_matrix

    # accuracy and confusion matrix calculated
    print("\nRandom Forest Classifier:")
    print("Accuracy percentage: ", accuracy_score(y_test, Y_pred) * 100, "%")
    print("Confusion Matrix\n", confusion_matrix(y_test.argmax(axis=1), Y_pred.argmax(axis=1)))
    PrecisionAndRecallRF(y_test, Y_pred)


def learning_curve_RF(classifier, X_train, X_test, y_train, y_test, training_size):
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=5, scoring='accuracy',
                                                            n_jobs=-1, train_sizes=training_size)
    train_mean = np.mean(train_scores, axis=1)
    x1 = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    x2 = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color="red", label="Training values")
    plt.plot(train_sizes, test_mean, '--', color="blue", label="Cross-validation values")

    plt.title("Learning Curve for Random Forest")
    plt.xlabel("Size of Training set")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def PrecisionAndRecallSVM(y_test, Y_pred):
    """
    :param y_test:
    :param Y_pred:
    :return:

    Calculates the Precision, Recall, F1-Score and Support for SVM classifier
    """
    print("Precision and Recall for SVM: \n", classification_report(y_test, Y_pred))

def PrecisionAndRecallRF(y_test, Y_pred):
    """
    :param y_test:
    :param Y_pred:
    :return:
    Calculates the Precision, Recall, F1-Score and Support for SVM classifier
    """
    print("Precision and Recall for SVM: \n", classification_report(y_test.argmax(axis=1), Y_pred.argmax(axis=1),
                                                                    target_names=['Data Science', 'Fitness', 'GOT']))


def main():
    """
    Read the data from json files
    """
    dataset1 = pd.read_json('DataScience.json', orient='split')
    dataset2 = pd.read_json('Fitness.json', orient='split')
    dataset3 = pd.read_json('GOT.json', orient='split')
    ds = [dataset1, dataset2, dataset3]
    # COncatenate all the data into one single set
    dataset = pd.concat(ds)
    Cleaning(dataset)


if __name__ == '__main__':
    main()


