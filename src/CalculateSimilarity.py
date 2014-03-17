'''
calculation of similarities is based on VSM.
(thus cosine similarity)
'''
import math
from scipy.sparse import csr_matrix

__author__ = 'kensk8er'

from utils.util import *


if __name__ == '__main__':
    # load data
    print 'load train data...'
    X_train = unpickle('data/train_tfidf.pkl')
    train_len, train_feature_len = X_train.get_shape()

    print 'load test data...'
    X_test = unpickle('data/test_tfidf.pkl')
    test_len, test_feature_len = X_test.get_shape()

    print 'reshape the matrix...'
    feature_len = max(train_feature_len, test_feature_len)
    X_train = csr_matrix((X_train.data, X_train.indices, X_train.indptr), shape=(train_len ,feature_len))
    X_test = csr_matrix((X_test.data, X_test.indices, X_test.indptr), shape=(test_len, feature_len))

    # iterate over every train document
    print 'calculate inner products between documents...'
    inner_products = X_train.dot(X_test.transpose())
    print 'calculate norm for train data...'
    train_norm = csr_matrix(X_train.multiply(X_train).sum(1)).sqrt()
    print 'calculate norm for test data...'
    test_norm = csr_matrix(X_test.multiply(X_test).sum(1)).sqrt()
    print 'calculate norm matrix...'
    norm_matrix = train_norm.dot(test_norm.transpose())
    print 'calculate cosine similarities...'
    similarities = inner_products / norm_matrix

    # save similarity
    print 'saving similarities...'
    enpickle(similarities, 'data/similarity.pkl')
