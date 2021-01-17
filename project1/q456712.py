from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
from joblib import Memory
from shutil import rmtree
from tempfile import mkdtemp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
import nltk
import random
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from matplotlib import pyplot as plt
from nltk import pos_tag
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from statistics import mean

np.random.seed(42)
random.seed(42)

TARGET_MAP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
}

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']

# load data and convert to 2 category
twenty_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=None)
twenty_train.target = np.array(
    list(map(lambda label: TARGET_MAP[label], twenty_train.target)))

twenty_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=None)
twenty_test.target = np.array(
    list(map(lambda label: TARGET_MAP[label], twenty_test.target)))


# feature extraction
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
wnl = nltk.wordnet.WordNetLemmatizer()

# configure stop words
stop_words = CountVectorizer(stop_words='english').get_stop_words()


def penn2morphy(penntag):
    # Converts Penn Treebank tags to WordNet
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    if len(penntag) > 1 and penntag[:2] in morphy_tag:
        return morphy_tag[penntag[:2]]
    else:
        return "n"


def lemmatize_sent(list_word):
    # Text input is list of strings, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(list_word)]


def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(CountVectorizer().build_analyzer()(doc)) if word not in stop_words and not word.isdigit())


def plot_roc(figure_name, fpr, tpr):
    # roc curve plot
    fig, ax = plt.subplots()

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

    fig.savefig(figure_name)


def fit_predict_and_plot_roc(figure_name, pipe, train_data, train_label, test_data, test_label):
    # fit, predict and plot roc
    pipe.fit(train_data, train_label)

    if hasattr(pipe, 'decision_function'):
        prob_score = pipe.decision_function(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score)
    else:
        prob_score = pipe.predict_proba(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score[:, 1])

    plot_roc(figure_name, fpr, tpr)

# ====== Q4 alternate approach (redo preprocessing every time) ======
# # r = 1000
# svm1000_pipline = Pipeline([
#     ('vect', CountVectorizer(min_df=3, analyzer=stem_rmv_punc)),
#     ('tfidf', TfidfTransformer()),
#     ('reduce_dim', TruncatedSVD(n_components=50)),
#     ('clf', LinearSVC(C=1000)),
# ])

# # roc curve
# fit_predict_and_plot_roc('q4_roc_r1000.png', svm1000_pipline, twenty_train.data,
#                          twenty_train.target, twenty_test.data, twenty_test.target)

# # confusion matrix
# test_data = svm1000_pipline["vect"].transform(twenty_test.data)
# test_data = svm1000_pipline["tfidf"].transform(test_data)
# test_data = svm1000_pipline["reduce_dim"].transform(test_data)

# plot_confusion_matrix(svm1000_pipline["clf"], test_data, twenty_test.target, display_labels=["computer technology", "recreational activity"], normalize="true")
# plt.savefig('q4_confusion_matrix_r1000.png')

# # r = 0.0001
# svm0001_pipline = Pipeline([
#     ('vect', CountVectorizer(min_df=3, analyzer=stem_rmv_punc)),
#     ('tfidf', TfidfTransformer()),
#     ('reduce_dim', TruncatedSVD(n_components=50)),
#     ('clf', LinearSVC(C=0.0001)),
# ])

# # roc curve
# fit_predict_and_plot_roc('q4_roc_r0001.png', svm0001_pipline, twenty_train.data,
#                          twenty_train.target, twenty_test.data, twenty_test.target)

# # confusion matrix
# test_data = svm0001_pipline["vect"].transform(twenty_test.data)
# test_data = svm0001_pipline["tfidf"].transform(test_data)
# test_data = svm0001_pipline["reduce_dim"].transform(test_data)


# plot_confusion_matrix(svm0001_pipline["clf"], test_data, twenty_test.target, display_labels=["computer technology", "recreational activity"], normalize="true")
# plt.savefig('q4_confusion_matrix_r0001.png')

# ====== end alternate approach ======

# Q4
# preprocessing
LSI_preprocessing_pipline = Pipeline([
    ('vect', CountVectorizer(min_df=3, analyzer=stem_rmv_punc)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50)),
])

reduced_train = LSI_preprocessing_pipline.fit_transform(twenty_train.data)
reduced_test = LSI_preprocessing_pipline.transform(twenty_test.data)

# r = 1000
svm1000_pipline = Pipeline([
    ('clf', LinearSVC(C=1000)),
])

# roc curve
fit_predict_and_plot_roc('q4_roc_r1000.png', svm1000_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(svm1000_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q4_confusion_matrix_r1000.png')

# r = 0.0001
svm00001_pipline = Pipeline([
    ('clf', LinearSVC(C=0.0001)),
])

# roc curve
fit_predict_and_plot_roc('q4_roc_r0001.png', svm00001_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(svm00001_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q4_confusion_matrix_r0001.png')

# TODO
# 1. calculate the accuracy, recall, precision and F-1 score of both SVM classifier. Which one performs better?
# 2. What happens for the soft margin SVM? Why is the case?


# cross validation
tradeoff = {1000: 0, 100: 0, 10: 0, 1: 0, 0.1: 0, 0.01: 0, 0.001: 0}
for c in tradeoff:
    clf = LinearSVC(C=c)
    fold_score = cross_val_score(clf, reduced_train, twenty_train.target, cv=5)
    tradeoff[c] = mean(fold_score)

# pipeline for the best tradeoff value
svm_C = max(tradeoff, key=tradeoff.get)
svm_best_pipline = Pipeline([
    ('clf', LinearSVC(C=svm_C)),
])

# roc curve
fit_predict_and_plot_roc('q4_roc_best.png', svm_best_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(svm_best_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q4_confusion_matrix_best.png')

# TODO
# calculate the accuracy, recall precision and F-1 score of this best SVM.


# Q5
# Note: the instruction "you may need to come up with some way toapproximate this if you use sklearn.linear model.LogisticRegression"
#       suggest using L2 and very large C to cancel the effect of regularization. But it seem to be outdated, now there is an None option.

# Logistic regression
logistic_pipline = Pipeline([
    ('clf', LogisticRegression(penalty="none")),
])
# roc curve
fit_predict_and_plot_roc('q5_roc_none.png', logistic_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)
# confusion matrix
plot_confusion_matrix(logistic_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q5_confusion_matrix_none.png')

# TODO
#  calculate the accuracy, recall precision and F-1 score of this classifier.

# cross validation on L1
tradeoff = {1000: 0, 100: 0, 10: 0, 1: 0, 0.1: 0, 0.01: 0, 0.001: 0}
for c in tradeoff:
    clf = LogisticRegression(C=c, penalty="l1", solver="liblinear")
    fold_score = cross_val_score(clf, reduced_train, twenty_train.target, cv=5)
    tradeoff[c] = mean(fold_score)

# pipeline for the best L1
log1_C = max(tradeoff, key=tradeoff.get)
log_bestl1_pipline = Pipeline([
    ('clf', LogisticRegression(C=log1_C, penalty="l1", solver="liblinear")),
])

# roc curve
fit_predict_and_plot_roc('q5_roc_bestL1.png', log_bestl1_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(log_bestl1_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q5_confusion_matrix_bestL1.png')

# cross validation on L2
tradeoff = {1000: 0, 100: 0, 10: 0, 1: 0, 0.1: 0, 0.01: 0, 0.001: 0}
for c in tradeoff:
    clf = LogisticRegression(C=c, penalty="l2")
    fold_score = cross_val_score(clf, reduced_train, twenty_train.target, cv=5)
    tradeoff[c] = mean(fold_score)

# pipeline for the best L2
log2_C = max(tradeoff, key=tradeoff.get)
log_bestl2_pipline = Pipeline([
    ('clf', LogisticRegression(C=log2_C, penalty="l2")),
])

# roc curve
fit_predict_and_plot_roc('q5_roc_bestL2.png', log_bestl2_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(log_bestl2_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q5_confusion_matrix_bestL2.png')

# TODO
# 1. Compare the performance (accuracy, precision, recall and F-1 score) of 3 logistic classifiers:
# w/o regularization, w/ L1 regularization and w/ L2 regularization (with the best
# parameters you found from the part above), using test data.
# 2. How does the regularization parameter affect the test error? How are the learnt coefficients
# affected? Why might one be interested in each type of regularization?
# 3. Both logistic regression and linear SVM are trying to classify data points using a linear
# decision boundary, then what’s the difference between their ways to find this boundary?
# Why their performance differ?


# Q6

# Naive Bayes
bayes_pipline = Pipeline([
    ('clf', GaussianNB()),
])

# roc curve
fit_predict_and_plot_roc('q6_roc.png', bayes_pipline, reduced_train,
                         twenty_train.target, reduced_test, twenty_test.target)

# confusion matrix
plot_confusion_matrix(bayes_pipline["clf"], reduced_test, twenty_test.target, display_labels=[
                      "computer technology", "recreational activity"], normalize="true")
plt.savefig('q6_confusion_matrix.png')

# TODO
#  calculate the accuracy, recall, precision and F-1 score of this classifier.


# Q7

# TODO
# remove “headers” and “footers” vs not


def stem_rmv_punc_nonlemmatize(doc):
    return (word for word in CountVectorizer().build_analyzer()(doc) if word not in stop_words and not word.isdigit())


cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

grid_pipline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50)),
    ('clf', GaussianNB()),
],
    memory=memory
)

param_grid = [
    {
        'vect__min_df': [3, 5],
        'vect__analyzer': [stem_rmv_punc, stem_rmv_punc_nonlemmatize],
        'reduce_dim': [TruncatedSVD(), NMF()],
        'reduce_dim__n_components': [50],
        'clf': [LinearSVC(C=svm_C)],
        'clf__C': [svm_C]
    },
    {
        'vect__min_df': [3, 5],
        'vect__analyzer': [stem_rmv_punc, stem_rmv_punc_nonlemmatize],
        'reduce_dim': [TruncatedSVD(), NMF()],
        'reduce_dim__n_components': [50],
        'clf': [LogisticRegression()],
        'clf__C': [log1_C],
        'clf__penalty': ["l1"],
        'clf__solver': ["liblinear"]
    },
    {
        'vect__min_df': [3, 5],
        'vect__analyzer': [stem_rmv_punc, stem_rmv_punc_nonlemmatize],
        'reduce_dim': [TruncatedSVD(), NMF()],
        'reduce_dim__n_components': [50],
        'clf': [LogisticRegression()],
        'clf__C': [log2_C],
        'clf__penalty': ["l2"],
    },
    {
        'vect__min_df': [3, 5],
        'vect__analyzer': [stem_rmv_punc, stem_rmv_punc_nonlemmatize],
        'reduce_dim': [TruncatedSVD(), NMF()],
        'reduce_dim__n_components': [50],
        'clf': [GaussianNB()],
    },
]

grid = GridSearchCV(grid_pipline, cv=5, n_jobs=1,
                    param_grid=param_grid, scoring='accuracy')
grid.fit(twenty_train.data, twenty_train.target)
rmtree(cachedir)

df = pd.DataFrame(grid.cv_results_)
df.to_csv('q7csv')

# TODO
# Q8-11

# Q12
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'misc.forsale', 'soc.religion.christian']
# TODO: decide remove header & footer or not
twenty_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=None)

twenty_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=None)

reduced_train = LSI_preprocessing_pipline.fit_transform(twenty_train.data)
reduced_test = LSI_preprocessing_pipline.transform(twenty_test.data)

# Naive Bayes
clf = GaussianNB().fit(reduced_train, twenty_train.target)

# confusion matrix
plot_confusion_matrix(clf, reduced_test, twenty_test.target,
                      display_labels=categories, normalize="true")
plt.savefig('q12_confusion_matrix_NB.png')
# TODO
#  calculate the accuracy, recall, precision and F-1 score

#  multiclass SVM classification
# one vs one
clf = OneVsOneClassifier(LinearSVC()).fit(reduced_train, twenty_train.target)
plot_confusion_matrix(clf, reduced_test, twenty_test.target,
                      display_labels=categories, normalize="true")
plt.savefig('q12_confusion_matrix_svm1to1.png')
# TODO
#  calculate the accuracy, recall, precision and F-1 score


# one vs rest
clf = OneVsRestClassifier(LinearSVC()).fit(reduced_train, twenty_train.target)
plot_confusion_matrix(clf, reduced_test, twenty_test.target,
                      display_labels=categories, normalize="true")
plt.savefig('q12_confusion_matrix_svm1to1.png')
# TODO
#  calculate the accuracy, recall, precision and F-1 score
