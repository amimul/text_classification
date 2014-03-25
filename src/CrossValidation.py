from collections import defaultdict
from scipy.sparse import csr_matrix
import sys

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


def calculate_similarity(docs1, docs2, N, k):
    docs_len = docs2.shape[0]

    # convert train document vector into csc format
    print 'transpose document vectors...'
    docs1 = docs1.transpose()

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
        docs1_target = docs1[:, list(doc1_ids)]
        similarity = doc2.dot(docs1_target)

        # sort by similarity
        pairs = zip(similarity.todense().tolist()[0], list(doc1_ids))
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
    neighbors_num = 100  # keep only k-nearest neighbors
    valid_num = 20000

    if sys.argv[1] == 'similarity':
        ## similarity calculation part (preprocessing for cross validation)

        # calculate partially and combine later
        print 'load document matrix...'
        X_train = unpickle('data/train_tfidf.pkl')
        X_test = X_train[-valid_num:, :]
        X_train = X_train[: -valid_num, :]
        train_len = X_train.shape[0]
        feature_len = X_train.shape[1]
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
        similarities = calculate_similarity(X_train, X_test, approximation_num, neighbors_num)

        # save similarities
        print 'save similarities...'
        enpickle(similarities, 'data/similarities_valid.pkl')

    elif sys.argv[1] == 'validation':
        ## classification part (cross validation)

        print 'load data...'
        similarities = unpickle('data/similarities_valid.pkl')
        Y_train = unpickle('data/train_label.pkl')
        Y_test = Y_train[-valid_num:]
        test_len = len(similarities)
        average_category_num = 0

        for k in range(10, 11):
            for alpha in range(50, 70, 5):
                # hyper parameters
                alpha /= float(100)
                default_category = 24177
                threshold = 0.1

                print 'k value:', k
                print 'alpha value:', alpha

                predicted = {}

                # iterate over every test document
                progress = 0
                for test_doc_id in xrange(test_len):
                    # print progress
                    if (float(test_doc_id) / test_len) * 100 > progress:
                        print '\r', progress, '%',
                        progress += 1

                    pairs = similarities[test_doc_id]

                    if max(pairs)[0] > 0:  # following procedures are only meaningful when there's non-zero similarity train document
                        pairs.sort()
                        pairs.reverse()
                        if len(pairs) > k:
                            pairs = pairs[0: k]  # keep only k-nearest neighbors

                        # calculate category scores
                        scores = defaultdict(int)
                        for pair in pairs:
                            similarity = pair[0]
                            train_doc_id = pair[1]
                            categories = Y_train[train_doc_id]

                            for category in categories:
                                category = int(category)
                                scores[category] += similarity  # this algorithm might have room to improve

                        # sort by descending order
                        scores = scores.items()
                        scores.sort(key=lambda entry: entry[1])
                        scores.reverse()
                        max_score = scores[0][1]

                        # choose top-x categories based on alpha value
                        categories = []
                        for score_tuple in scores:
                            category = score_tuple[0]
                            score = score_tuple[1]

                            if max_score > 0:  # some test data don't have any feature so max_score could be zero
                                if score / max_score > alpha:
                                    categories.append(category)
                            else:
                                categories = [default_category]  # TBF: room to improve
                                break

                    else:  # procedures when there isn't any non-zero similarity train document
                        categories = [default_category]  # TBF: room to improve

                    # insert categories into predictions
                    predicted[test_doc_id] = categories
                    average_category_num += len(predicted[test_doc_id])

                ## calculate Macro F1 Score
                correct_docs = {}  # key: category, value: doc_ids
                predicted_docs = {}  # key: category, value: doc_ids
                average_category_num /= float(test_len)

                # generate correct documents for each category
                for test_doc_id in xrange(test_len):
                    categories = Y_test[test_doc_id]

                    for category in categories:
                        category = int(category)
                        if not correct_docs.has_key(category):
                            correct_docs[category] = set([test_doc_id])
                        else:
                            correct_docs[category] |= {test_doc_id}

                # generate predicted documents for each category
                for test_doc_id in xrange(test_len):
                    categories = predicted[test_doc_id]

                    for category in categories:
                        category = int(category)
                        if not predicted_docs.has_key(category):
                            predicted_docs[category] = set([test_doc_id])
                        else:
                            predicted_docs[category] |= {test_doc_id}

                # calculate Macro Precision
                correct_category_num = len(correct_docs)
                macro_precision = 0
                for category in correct_docs.keys():
                    if not predicted_docs.has_key(category):
                        predicted_docs[category] = set()
                        macro_precision += 0  # add zero for the case when no document is retrieved
                    else:
                        macro_precision += float(len(correct_docs[category] & predicted_docs[category])) / len(predicted_docs[category])
                macro_precision /= correct_category_num

                # calculate Macro Recall
                macro_recall = 0
                for category in correct_docs.keys():
                    if not predicted_docs.has_key(category):
                        predicted_docs[category] = set()
                    macro_recall += float(len(correct_docs[category] & predicted_docs[category]) / len(correct_docs[category]))
                macro_recall /= correct_category_num

                # calculate MaF
                print '\rMacro Precision:', macro_precision
                print 'Macro Recall:', macro_recall
                MaF = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
                print 'Macro F1-score:', MaF
                print 'average number of predicted categories:', average_category_num
                print ''

        print 'done!'

    else:
        print 'input arguments!'
