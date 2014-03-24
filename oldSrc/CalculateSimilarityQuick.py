__author__ = 'kensk8er'

from utils.util import *
import numpy as np
from numpy import matrix

if __name__ == '__main__':

    # hyper parameter
    N = 30  # find N most high entries. increase this to speed up the computation.

    # calculate partially and combine later
    print 'load document matrix...'
    X_train = unpickle('data/train_svd_new50.pkl')
    X_train = matrix(X_train)
    train_len = X_train.shape[0]
    X_test = unpickle('data/test_svd_new50.pkl')
    X_test = matrix(X_test)
    test_len = X_test.shape[0]

    print 'find high value entries from test data...'
    X_test_order = X_test.argsort()
    feature_len = X_test.shape[1]

    # calculate norms
    print 'calculate norms...'
    train_norm = np.sqrt(np.multiply(X_train, X_train).sum(axis=1))
    test_norm = np.sqrt(np.multiply(X_test, X_test).sum(axis=1))

    # cache feature-wise promising train documents ids information
    promising_documents = [0 for i in xrange(feature_len)]
    print 'cache promising train documents for each features...'
    for feature_id in xrange(feature_len):
        feature_mean = X_train[:, feature_id].mean()
        feature_std = X_train[:, feature_id].std()
        feature_max = X_train[:, feature_id].max()
        if (feature_mean + feature_std * 2) < feature_max:
            threshold = feature_mean + feature_std  # retrieve top 64% or so * this may be too hash criteria
        else:
            threshold = feature_mean  # TBF: still room to improve
        promising_documents[feature_id] = X_train[:, feature_id] > threshold

    # normalize document vectors
    print 'normalize document vectors...'
    X_train = X_train / train_norm
    X_test = X_test / test_norm

    # iterate over every test document
    similarities = {}  # key: int test_doc_id, value: tuple(list[int] train_docs, list[float] similarity_values)
    progress = 0
    print 'calculate similarities...'
    for test_doc_id in xrange(test_len):
        # print progress
        if (float(test_doc_id) / test_len) * 100 > progress:
            progress = float(test_doc_id) / test_len * 100
            print '\r', progress, '%',

        #train_doc_ids = matrix(0)
        #previous_doc_ids = matrix(0)
        train_doc_ids = promising_documents[X_test_order[test_doc_id, -1]]
        for i in range(2, N):
            if train_doc_ids.sum() > 30:
                train_doc_ids = np.multiply(train_doc_ids, promising_documents[X_test_order[test_doc_id, -i]])
            else:
                break

        # first_attempt = True
        # for position in range(1, N + 1):
        #     feature_id = X_test_order[test_doc_id, -position]
        #
        #     # find the train documents to compute similarity
        #     if first_attempt is True:
        #         train_doc_ids = promising_documents[feature_id]
        #     else:
        #         # possibly it's better to do addition instead multiplication, but couldn't be helped because of computational complexity
        #         train_doc_ids = np.multiply(train_doc_ids, promising_documents[feature_id])
        #
        #     if train_doc_ids.sum() < 1000:  # cause error if there's only one train document retrieved
        #         if previous_doc_ids.sum() > 0:
        #             train_doc_ids = previous_doc_ids
        #             break
        #     else:
        #         previous_doc_ids = train_doc_ids
        #         first_attempt = False

        # calculate similarities
        train_doc_ids = np.where(train_doc_ids == 1)[0]
        test_doc = X_test[test_doc_id, :]  # maybe this copy process slows down the program?
        train_docs = X_train[train_doc_ids, :][0]  # maybe this copy process slows down the program?
        similarity = test_doc * train_docs.transpose()
        similarities[test_doc_id] = (train_doc_ids.tolist(), similarity.tolist())
    print '\r100% done!'

    # save similarities
    print 'save similarities...'
    enpickle(similarities, 'data/similarities.pkl')
