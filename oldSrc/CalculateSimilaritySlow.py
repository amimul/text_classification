"""
calculation of similarities is based on VSM.
(thus cosine similarity)
"""
import math
from scipy.sparse import csr_matrix
import numpy as np
import cProfile

__author__ = 'kensk8er'

from utils.util import *


if __name__ == '__main__':
    # load data
    print 'load train data...'
    X_train = unpickle('data/train_min_tfidf.pkl')
    train_len, train_feature_len = X_train.get_shape()

    print 'load test data...'
    X_test = unpickle('data/test_min_tfidf.pkl')
    test_len, test_feature_len = X_test.get_shape()

    print 'reshape the matrix...'
    feature_len = max(train_feature_len, test_feature_len)
    X_train = csr_matrix((X_train.data, X_train.indices, X_train.indptr), shape=(train_len, feature_len))
    X_test = csr_matrix((X_test.data, X_test.indices, X_test.indptr), shape=(test_len, feature_len))

    # iterate over every train document (because it causes an error when there are two many non-zero elements)
    similarities = np.array([])
    progress = 0
    print 'calculate similarities...'
    for train_doc_id in xrange(train_len):
        # print progress
        if (float(train_doc_id) / train_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        train_row = X_train.getrow(train_doc_id)  # 1 * feature_len matrix
        inner_products = train_row.dot(X_test.transpose())  # 1 * test_len matrix
        train_norm = csr_matrix(train_row.multiply(train_row).sum(1)).sqrt()  # 1 * 1 matrix
        test_norm = csr_matrix(X_test.multiply(X_test).sum(1)).sqrt()  # test_len * 1 matrix
        norm_matrix = train_norm.dot(test_norm.transpose())  # 1 * test_len matrix
        similarity = inner_products / norm_matrix  # 1 * test_len matrix

        if similarities.size > 0:
            # TBF: use scipy.sparse.vstack instead
            similarities = np.vstack((similarities, similarity.todense()))
        else:
            similarities = similarity.todense()

    similarities = csr_matrix(similarities)
    print '\r100 % done!'

    # save similarity
    print 'saving similarities...'
    enpickle(similarities, 'data/similarity_min.pkl')
