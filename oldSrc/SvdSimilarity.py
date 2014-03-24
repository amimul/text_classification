"""
perform SVD on tfidf matrix and calculate similarity using VSM.
"""
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.linalg import diagsvd
import scipy.sparse as sp
from numpy import matrix
import numpy as np
import sys

__author__ = 'kensk8er'

from utils.util import *
import numpy as np


def main():
    # hyper parameters
    dimension = 5#50

    print 'load data...'
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
    U, Sigma, V = linalg.svds(X, k=dimension, which='LM', tol=0.1)
    print 'comprise matrix...'
    U = matrix(U)  # doc_len * dimension
    S = matrix(diagsvd(Sigma, dimension, dimension))  # dimension * dimension
    #V = matrix(V[:, :dimension])  # dimension * dimension
    X = U * S  # doc_len * dimension
    X_train, X_test, c = np.vsplit(X, np.array([train_len, train_len + test_len]))
    X_train = matrix(X_train)
    X_test = matrix(X_test)

    print 'save document matrix...'
    enpickle(X_train, 'data/train_svd.pkl')
    enpickle(X_test, 'data/test_svd.pkl')

    ## this is for vectorized way of calculation (cause memory error for huge matrix)
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
    enpickle(similarities, 'data/similarity_svd.pkl')

    print 'done!'


def calculate_similarity():
    # matrix implementation
    threshold = 0.5  # keep only values which similarity is more than the threshold

    print 'load document matrix...'
    X_train = unpickle('data/train_svd.pkl')
    train_len = X_train.shape[0]
    train_len /= 10
    X_test = unpickle('data/test_svd.pkl')
    test_len = X_test.shape[0]

    # iterate over every train document (because it causes an error when there are two many non-zero elements)
    similarities = csr_matrix(0)
    progress = 0
    batch_size = 1000
    residual = train_len % batch_size
    quotient = train_len // batch_size
    #test_memory = np.array([[0 for i in xrange(batch_size)] for j in xrange(test_len)])
    print 'calculate similarities...'
    test_norm = np.sqrt(np.multiply(X_test, X_test).sum(axis=1))  # test_len * 1 matrix
    for train_doc_id in xrange(0, train_len, batch_size):
        # print progress
        if (float(train_doc_id) / train_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        train_batch = X_train[train_doc_id: train_doc_id + batch_size, :]  # batch_size * feature_len matrix
        inner_products = train_batch * X_test.transpose()  # batch_size * test_len matrix
        train_norm = np.sqrt(np.multiply(train_batch, train_batch).sum(axis=1))  # batch_size * 1 matrix
        norm_matrix = train_norm * test_norm.transpose()  # batch_size * test_len matrix
        similarity = inner_products / norm_matrix  # batch_size * test_len matrix
        # convert into sparse matrix
        similarity = np.multiply(similarity > threshold, similarity)

        if similarities.size > 0:
            similarities = sp.vstack([similarities, similarity], format='csr')
        else:
            similarities = csr_matrix(similarity)

    # residual
    if residual > 0:
        train_doc_id = batch_size * quotient
        train_batch = X_train[train_doc_id: train_doc_id + batch_size, :]  # batch_size * feature_len matrix
        inner_products = train_batch * X_test.transpose()  # batch_size * test_len matrix
        train_norm = np.sqrt(np.multiply(train_batch, train_batch).sum(axis=1))  # batch_size * 1 matrix
        norm_matrix = train_norm * test_norm.transpose()  # batch_size * test_len matrix
        similarity = inner_products / norm_matrix  # batch_size * test_len matrix
        # convert into sparse matrix
        similarity = np.multiply(similarity > threshold, similarity)

        similarities = sp.vstack([similarities, similarity], format='csr')

    print '\r100 % done!'

    # save similarity
    print 'saving similarities...'
    enpickle(similarities, 'data/similarity_svd.pkl')

    print 'done!'


def calculate_similarity_enhanced():
    # calculate partially and combine later
    print 'load document matrix...'
    X_train = unpickle('data/train_svd.pkl')
    train_len = X_train.shape[0]
    X_test = unpickle('data/test_svd.pkl')
    test_len = X_test.shape[0]

    # iterate over every train document (because it causes an error when there are two many non-zero elements)
    progress = 0
    print 'calculate norms...'
    train_norm = np.sqrt(np.multiply(X_train, X_train).sum(axis=1))  # train_len * 1 matrix
    test_norm = np.sqrt(np.multiply(X_test, X_test).sum(axis=1))  # test_len * 1 matrix

    # parameter
    batch_size = 10000
    train_residual = train_len % batch_size
    train_quotient = train_len // batch_size
    test_residual = test_len % batch_size
    test_quotient = test_len // batch_size

    train_batch = {}
    progress = 0
    print 'calculate inner products...'
    for test_doc_idx in xrange(test_quotient):
        # print progress
        if (float(test_doc_idx) / test_quotient) * 100 > progress:
            progress = int((float(test_doc_idx) / test_quotient) * 100)
            print '\r', progress, '%',

        test_batch = X_test[test_doc_idx * batch_size: (test_doc_idx + 1) * batch_size, :]
        inner_products = matrix([])

        for train_doc_idx in xrange(train_quotient):
            # initialize at the first loop
            if test_doc_idx == 0:
                train_batch[train_doc_idx] = X_train[train_doc_idx * batch_size: (train_doc_idx + 1) * batch_size, :]

            # calculate inner products
            if inner_products.size > 0:

                inner_products = sp.vstack((inner_products, train_batch[train_doc_idx] * test_batch.transpose()))
            else:
                inner_products = train_batch[train_doc_idx] * test_batch.transpose()
            print '.',

        # residual
        if train_residual > 0:
            # initialize at the first loop
            if test_doc_idx == 0:
                train_batch[train_quotient] = X_train[train_quotient * batch_size: train_quotient * batch_size + train_residual, :]
            inner_products = np.vstack((inner_products, train_batch[train_quotient] * test_batch.transpose()))

        # record similarity
        norm_matrix = train_norm * test_norm[test_doc_idx * batch_size: (test_doc_idx + 1) * batch_size, :].transpose()
        similarity = inner_products / norm_matrix
        file_name = "data/similarity_%s.pkl" % test_doc_idx
        enpickle(similarity, file_name)

    #residual
    if test_residual > 0:
        inner_products = matrix([])
        test_batch = X_test[test_quotient * batch_size: test_quotient * batch_size + test_residual, :]

        for train_doc_idx in xrange(train_quotient):
            # initialize at the first loop
            train_batch[train_doc_idx] = X_train[train_doc_idx * batch_size: (train_doc_idx + 1) * batch_size, :]

            # calculate inner products
            if inner_products.size > 0:
                inner_products = np.vstack((inner_products, train_batch[train_doc_idx] * test_batch.transpose()))
            else:
                inner_products = train_batch[train_doc_idx] * test_batch.transpose()

        # residual
        if train_residual > 0:
            inner_products = np.vstack((inner_products, train_batch[train_quotient] * test_batch.transpose()))

        # record similarity
        norm_matrix = train_norm * test_norm[test_quotient * batch_size: (test_quotient + 1) * batch_size, :].transpose()
        similarity = inner_products / norm_matrix
        file_name = "data/similarity_%s.pkl" % test_quotient
        enpickle(similarity, file_name)

    print '100 % done!'


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'similarity':
        calculate_similarity()
    elif sys.argv[1] == 'enhanced':
        calculate_similarity_enhanced()

