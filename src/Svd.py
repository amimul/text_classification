"""
perform SVD on tfidf matrix
"""
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.linalg import diagsvd
from numpy import matrix

__author__ = 'kensk8er'

from utils.util import *
import numpy as np

if __name__ == '__main__':
    # hyper parameters
    dimension = 200

    X_train = unpickle('data/train_tfidf.pkl')
    train_len, train_feature_len = X_train.get_shape()
    X_test = unpickle('data/test_tfidf.pkl')
    test_len, test_feature_len = X_test.get_shape()

    print 'reshape the matrix...'
    feature_len = max(train_feature_len, test_feature_len)
    X_train = csr_matrix((X_train.data, X_train.indices, X_train.indptr), shape=(train_len, feature_len))
    X_test = csr_matrix((X_test.data, X_test.indices, X_test.indptr), shape=(test_len, feature_len))


    X = vstack((X_train, X_test), format='csr')

    # svd
    print 'perform svd...'
    U, Sigma, V = linalg.svds(X, k=dimension, which='LM')
    U = matrix(U)  # doc_len * dimension
    S = matrix(diagsvd(Sigma, dimension, dimension))  # dimension * dimension
    V = matrix(V[:, :dimension])  # dimension * dimension
    X = U * S  # doc_len * dimension
    X = X * V

    X_train, X_test, c = np.vsplit(X, np.array([train_len, train_len + test_len]))
    X_train = csr_matrix(X_train)
    X_test = csr_matrix(X_test)

    print 'save document matrix...'
    enpickle(X_train, 'data/train_svd.pkl')
    enpickle(X_test, 'data/test_svd.pkl')

    print 'done!'

