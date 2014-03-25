from sklearn.datasets import load_svmlight_file

__author__ = 'kensk8er'

from utils.util import *

if __name__ == '__main__':
    print 'load labels...'
    train_labels = unpickle('data/train_label.pkl')
    train_len = len(train_labels)

    print 'load hierarchies...'
    hierarchies = {}  # key: child, value: parents
    inverted_index = {}  # key: parent, value: children
    file = open('data/hierarchy.txt', 'r')
    for line in file:
        data = line.split(' ')
        child = int(data[1].rstrip())
        parent = int(data[0])

        if not hierarchies.has_key(child):
            hierarchies[child] = [parent]
        else:
            hierarchies[child].append(parent)

        if not inverted_index.has_key(parent):
            inverted_index[parent] = [child]
        else:
            inverted_index[parent].append(child)

    # iterate over every train data
    relatives = {}
    for train_doc_id in xrange(train_len):
        if train_doc_id % 1000 == 0:
            print '\r', float(train_doc_id) / train_len * 100, '%',

        labels = train_labels[train_doc_id]

        for label in labels:
            label = int(label)

            # do the below procedure only for once
            if not relatives.has_key(label):

                # retrieve parents
                if hierarchies.has_key(label):
                    parents = hierarchies[label]

                    # retrieve relatives for each parent
                    relatives[label] = set()
                    for parent in parents:
                        relatives[label] |= set(inverted_index[parent])

                    # discard itself
                    relatives[label].discard(label)
                    relatives[label] = list(relatives[label])
                else:
                    relatives[label] = []
    print '\r100 % done!'

    print 'save relatives...'
    enpickle(relatives, 'data/relatives.pkl')
