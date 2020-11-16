# Text training as described in https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# I also tried optimizing even further with the grid search
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np


def read_text_topic_file(filename):
    with open(filename, mode='r', encoding='utf-8') as train_file:
        csv_reader = csv.reader(train_file)
        header = next(csv_reader)  # skip the heading
        assert (header == ['text', 'topic', 'topic_label'])

        X_text, Y = [], []
        for train_line in csv_reader:
            text, topic = train_line[0], train_line[1]
            X_text.append(text)
            Y.append(topic)

    return X_text, Y


def run_classifier_pipeline(pipeline, X_train_text, Y_train, X_test_text, Y_test):
    pipeline.fit(X_train_text, Y_train)
    Y_predicted = pipeline.predict(X_test_text)
    return np.mean(Y_predicted == Y_test)


def main():
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    val_filename = 'val.csv'

    # Read in the training and test data
    X_train_text, Y_train = read_text_topic_file(train_filename)
    X_test_text, Y_test = read_text_topic_file(test_filename)

    # Classify with Naive Bayes
    nb_text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    print("naive bayes precision = ", run_classifier_pipeline(nb_text_clf, X_train_text, Y_train, X_test_text, Y_test))

    # Classify with SGD
    sgd_text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
    ])

    print("sgd precision = ", run_classifier_pipeline(sgd_text_clf, X_train_text, Y_train, X_test_text, Y_test))

    # Try adding in the validation data to the training since we don't need the validation data to tune
    # any hyper-parameters.
    X_val_text, Y_val = read_text_topic_file(val_filename)
    X_train_text.extend(X_val_text)
    Y_train.extend(Y_val)

    print("naive bayes precision (adding val data) = ",
          run_classifier_pipeline(nb_text_clf, X_train_text, Y_train, X_test_text, Y_test))
    print("sgd precision (adding val data) = ",
          run_classifier_pipeline(sgd_text_clf, X_train_text, Y_train, X_test_text, Y_test))


if __name__ == '__main__':
    main()
