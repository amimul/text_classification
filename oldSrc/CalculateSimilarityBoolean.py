from scipy.sparse import csr_matrix

__author__ = 'kensk8er'

from utils.util import *
import numpy as np


def cache_promising_documents(data):
    """
    cache feature-wise promising train document ids information

    :rtype : list
    :type data: csr_matrix
    :param data:
    data for which to return promising documents feature-wise
    """
    promising_documents = []
    data_csc = data.tocsc()
    feature_len = data.shape[1]

    print 'cache promising train documents for each features...'
    for feature_id in xrange(feature_len):
        # print progress
        progress = float(feature_id) / feature_len * 100
        print '\r', progress, '%',

        promising_documents.append(data_csc.getcol(feature_id).nonzero()[0])
    print '\r100% done!'

    return promising_documents


def normalize_document(data, norm):
    """
    normalize document vector by given norm matrix

    :type norm: matrix
    :param norm:
    :type data: csr_matrix
    :param data:
    :return:
    """
    ind_ptrs = data.indptr
    data_len = data.shape[0]
    for i in xrange(data_len):
        # print progress
        if i % 1000 == 0:
            progress = float(i) / data_len * 100
            print '\r', progress, '%',

        indices = range(ind_ptrs[i], ind_ptrs[i + 1])
        data.data[indices] /= norm[i, 0]
    print '\r100% done!'

    return data


def calculate_similarity(docs1, docs2, N, k, promising_documents):
    docs_len = docs2.shape[0]

    # convert train document vector into csc format
    print 'transpose document vectors...'
    #docs1 = docs1.transpose()

    # iterate over every test document
    similarities = {}
    print 'calculate similarities...'
    for doc2_id in xrange(docs_len):
        # print progress
        progress = float(doc2_id) / docs_len * 100
        print '\r', progress, '%',

        doc1_ids = set()
        doc2 = docs2.getrow(doc2_id)
        doc2_order = np.argsort(doc2.data)

        # find the train documents to compute similarity
        iterate_num = min(N, doc2.nnz)
        position = 1
        while position <= iterate_num:
            feature_id = doc2.nonzero()[1][doc2_order[-position]]
            doc1_ids |= set(promising_documents[feature_id])
            position += 1

        if len(doc1_ids) == 0:
            doc1_ids |= {0}  # add 0 if there isn't any promising document

        # calculate similarities
        # iterate over every doc1
        similarity = []
        for doc1_id in doc1_ids:
            doc1 = docs1.getrow(doc1_id)
            intercept = len(set(doc1.nonzero()[1]) & set(doc2.nonzero()[1]))
            similarity.append(1 - float(doc1.nnz + doc2.nnz - 2 * intercept) / (doc1.nnz + doc2.nnz - intercept))

        # sort by similarity
        pairs = zip(similarity, list(doc1_ids))
        pairs.sort()
        pairs.reverse()
        if len(pairs) > k:
            pairs = pairs[0: k]  # keep only k-nearest neighbors

        similarities[doc2_id] = pairs
    print '\r100% done!'

    return similarities


if __name__ == '__main__':

    # hyper parameters
    approximation_num = 3  # decrease this to speed up the computation.
    neighbors_num = 10  # keep only k-nearest neighbors

    # calculate partially and combine later
    print 'load document matrix...'
    X_train = unpickle('data/train_tfidf.pkl')
    train_len = X_train.shape[0]
    feature_len = X_train.shape[1]
    X_test = unpickle('data/test_tfidf.pkl')
    test_len = X_test.shape[0]

    # calculate norms
    print 'calculate norms...'
    train_norm = np.sqrt(X_train.multiply(X_train).sum(1))
    test_norm = np.sqrt(X_test.multiply(X_test).sum(1))

    # cache feature-wise promising documents
    promising_documents = cache_promising_documents(X_train)

    # normalize document vector
    print 'normalize train document vectors...'
    X_train = normalize_document(X_train, train_norm)
    print 'normalize test document vectors...'
    X_test = normalize_document(X_test, test_norm)

    # calculate similarity
    similarities = calculate_similarity(X_train, X_test, approximation_num, neighbors_num, promising_documents)

    # save similarities
    print 'save similarities...'
    enpickle(similarities, 'data/similarities.pkl')
