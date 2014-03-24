import math
from scipy.sparse import csr_matrix

__author__ = 'kensk8er'

from utils.util import *
import numpy as np

if __name__ == '__main__':
    print 'load train data...'
    X_train = unpickle('data/train_tfidf.pkl')
    train_len = X_train.shape[0]
    print 'load test data...'
    X_test = unpickle('data/test_tfidf.pkl')
    test_len = X_test.shape[0]
    print 'load df...'
    df = unpickle('data/df.pkl')

    percentage = 0.2

    # reduce dimension for train data
    base = 2  # base for log function
    progress = 0
    index = 0  # index of the sparse matrix to which update value
    print 'reduce dimension for train data...'
    for i in xrange(train_len):
        # print progress
        if (float(i) / train_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        row = X_train.getrow(i)
        row_order = np.argsort(row.data)
        top_k = int(math.ceil(float(row.nnz) * percentage))

        for j in range(len(row_order) - top_k):
            X_train.data[index + row_order[j]] = 0

        for j in range(len(row_order) - top_k, len(row_order)):
            term = row.nonzero()[1][row_order[j]]
            if not df[term] > 1:
                X_train.data[index + row_order[j]] = 0
        index += row.nnz
    X_train.eliminate_zeros()
    print '\r100 % done!'

    progress = 0
    index = 0  # index of the sparse matrix to which update value
    print 'reduce dimension for test data...'
    for i in xrange(test_len):
        # print progress
        if (float(i) / test_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        row = X_test.getrow(i)
        row_order = np.argsort(row.data)
        top_k = int(math.ceil(float(row.nnz) * percentage))

        for j in range(len(row_order) - top_k):
            X_test.data[index + row_order[j]] = 0

        for j in range(len(row_order) - top_k, len(row_order)):
            term = row.nonzero()[1][row_order[j]]
            if not df[term] > 1:
                X_test.data[index + row_order[j]] = 0
        index += row.nnz
    X_test.eliminate_zeros()
    print '\r100 % done!'

    print 'reshape the matrix...'
    feature_len = max(X_train.shape[1], X_test.shape[1])
    X_train = csr_matrix((X_train.data, X_train.indices, X_train.indptr), shape=(train_len ,feature_len))
    X_test = csr_matrix((X_test.data, X_test.indices, X_test.indptr), shape=(test_len, feature_len))

    print 'saving train data...'
    enpickle(X_train, 'data/train_tfidf_best.pkl')
    print 'done!'

    print 'saving test data...'
    enpickle(X_test, 'data/test_tfidf_best.pkl')
    print 'done!'

