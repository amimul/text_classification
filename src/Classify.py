"""
choose multiple categories based on similarities.
category x is chosen when: score(x) / score(top) > alpha (alpha is a constant between 0 and 1, top is class with the highest score)
score are calculated by looking at k-nearest neighbors
decide alpha by minimizing: error = average_num_of_categories_predicted - average_num_of_categories

hyper parameters: alpha, k
"""
from collections import defaultdict
import csv
from scipy.sparse import csr_matrix

__author__ = 'kensk8er'

from utils.util import *
import numpy as np

if __name__ == '__main__':
    print 'load data...'
    similarities = unpickle('data/similarity_min_svd.pkl')
    Y_train = unpickle('data/train_min_label.pkl')
    train_len = len(Y_train)
    Y_test = unpickle('data/test_min_label.pkl')
    test_len = len(Y_test)

    # hyper parameters
    alpha = 0.9
    k = 10
    predicted = {}

    # iterate over every test document
    progress = 0
    print 'classify categories for each test document...'
    for test_doc_id in xrange(test_len):  # test_doc_id starts from 0, thus you need to add 1 to this at the end
        # print progress
        if (float(test_doc_id) / test_len) * 100 > progress:
            print '\r', progress, '%',
            progress += 1

        # sort by similarity
        test_col = similarities[:, test_doc_id]
        if not all(test_col[:, 0] == 0):  # following procedures are only meaningful when there's non-zero similarity train document
            sort_indices = [i[0] for i in test_col.argsort(axis=0)[::-1].tolist()]
            sort_indices = sort_indices[0:k]  # keep only k-nearest neighbors
            sorted_pairs = [(test_col[i, 0], i) for i in sort_indices]

            # calculate category scores
            scores = defaultdict(int)
            for pair in sorted_pairs:
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

                if score / max_score > alpha:
                    categories.append(category)

        else:  # procedures when there isn't any non-zero similarity train document
            categories = [314523]  # TBF: room to improve

        # insert categories into predictions
        document_id = int(Y_test[test_doc_id][1])  # test_doc_id starts from zero so you can't directly use it
        predicted[document_id] = categories
    print '\r100 % done!'

    # write to csv file
    print 'save the result into csv file...'
    filename = 'result/result_min_svd.csv'
    writecsv = csv.writer(file(filename, 'w'), lineterminator='\n')
    writecsv.writerow(['Id', 'Predicted'])

    predicted = predicted.items()
    predicted.sort()

    for pred in predicted:
        document_id = pred[0]
        categories = " ".join(map(str,pred[1]))
        writecsv.writerow([document_id, categories])
