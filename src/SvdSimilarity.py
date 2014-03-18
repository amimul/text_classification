"""
perform SVD on tfidf matrix
"""
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.linalg import diagsvd
from numpy import matrix
import numpy as np

__author__ = 'kensk8er'

from utils.util import *
import numpy as np

if __name__ == '__main__':
    # hyper parameters
    dimension = 200

    print 'load data...'
    X_train = unpickle('data/train_min_tfidf.pkl')
    train_len, train_feature_len = X_train.get_shape()
    X_test = unpickle('data/test_min_tfidf.pkl')
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
    X_train, X_test, c = np.vsplit(X, np.array([train_len, train_len + test_len]))
    X_train = matrix(X_train)
    X_test = matrix(X_test)

    print 'save document matrix...'
    enpickle(X_train, 'data/train_min_svd.pkl')
    enpickle(X_test, 'data/test_min_svd.pkl')

    # calculate similarity
    print 'calculate inner products between documents...'
    inner_products = X_train * X_test.transpose()
    print 'calculate norm for train data...'
    train_norm = np.sqrt(np.multiply(X_train, X_train).sum(axis=1))
    print 'calculate norm for test data...'
    test_norm = np.sqrt(np.multiply(X_test, X_test).sum(axis=1))
    print 'calculate norm matrix...'
    norm_matrix = train_norm * test_norm.transpose()
    print 'calculate cosine similarities...'
    similarities = inner_products / norm_matrix

    # save similarity
    print 'saving similarities...'
    enpickle(similarities, 'data/similarity_min_svd.pkl')

    print 'done!'

