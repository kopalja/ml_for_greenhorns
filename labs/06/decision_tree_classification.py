#!/usr/bin/env python3

#dd7e3410-38c0-11e8-9b58-00505601122b
#6e14ef6b-3281-11e8-9de3-00505601122b

import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


import math



class Node:
    def __init__(self, feature_index = None, split = None, return_value = None, min_criterion = None):
        self.left = None
        self.right = None
        self.feature_index = feature_index
        self.split = split
        self.return_value = return_value
        self.min_criterion = min_criterion
        self.data = None
        self.targets = None

    def predict(self, d, depth):
        if self.return_value != None:
            return self.return_value
        elif d[self.feature_index] <= self.split:
            return self.left.predict(d, depth + 1)
        else:
            return self.right.predict(d, depth + 1)
        

    # def print_tree(self, depth):
    #     if self.return_value != None:
    #         print('Leaf. Return value {0}'.format(self.return_value))
    #     else:
    #         self.left.print_tree(depth + 1)
    #         print('depth: {0}, feature name: {1}, criterion: {2}, split {3}'.format(depth, labels[self.feature_index], self.min_criterion, self.split))
    #         self.right.print_tree(depth + 1)



class DecisionTree:
    def __init__(self, args):
        self._args = args
        self.loss_fn = { "gini": self._gini, "entropy": self._entropy }
    
    def build_tree(self, data, targets):
        if self._args.max_leaves is None:
            self._root = self._split_node(data, targets, depth = 0)
        else:
            self._root = Node(return_value = self._find_most_frequent_item(targets))
            self._root.data = data
            self._root.targets = targets
            self._open_nodes = [self._root]
            while len(self._open_nodes) < self._args.max_leaves:
                leaf_to_split = None
                min_criterion = 10000
                for open_node in self._open_nodes:
                    feature_index, split, criterion = self._find_best_feature_split(open_node.data, open_node.targets)
                    criterion = criterion - self.loss_fn[args.criterion](open_node.targets)
                    if criterion < min_criterion and open_node.data.shape[0] >= self._args.min_to_split:
                        min_criterion = criterion
                        leaf_to_split = open_node

                # no Leaf could be splitted
                if leaf_to_split == None:
                    return
                node = self._split_node(leaf_to_split.data, leaf_to_split.targets, 0, create_leaf = False, node = leaf_to_split)
                self._open_nodes.append(node.left)
                self._open_nodes.append(node.right)
                self._open_nodes.remove(leaf_to_split)        



    def predict(self, data):
        #self._root.print_tree(0)
        return [self._root.predict(d, 0) for d in data]




    def _split_node(self, data, targets, depth, create_leaf = False, node = None):
        if create_leaf:
            node = Node(return_value = self._find_most_frequent_item(targets))
            node.data = data
            node.targets = targets
            return node
        # don't split node => leaf
        if (self._args.max_depth != None and depth == self._args.max_depth) or data.shape[0] < self._args.min_to_split or \
             self.loss_fn[args.criterion](targets) == 0:
            return Node(return_value = self._find_most_frequent_item(targets))

        best_feature_index, best_split, min_criterion = self._find_best_feature_split(data, targets)

        # split data
        data_left, targets_left = zip(*[(d, t) for d, t in zip(data, targets) if d[best_feature_index] <= best_split])
        data_right, targets_right  = zip(*[(d, t) for d, t in zip(data, targets) if d[best_feature_index] > best_split])

        # create node
        if node is None:
            node = Node(feature_index = best_feature_index, split = best_split, min_criterion = min_criterion)
            node.left = self._split_node(np.array(data_left), np.array(targets_left), depth + 1)
            node.right = self._split_node(np.array(data_right), np.array(targets_right), depth + 1)
        else:
            node.feature_index = best_feature_index
            node.split = best_split
            node.min_criterion = min_criterion
            node.return_value = None
            node.left = self._split_node(np.array(data_left), np.array(targets_left), depth + 1, create_leaf = True)
            node.right = self._split_node(np.array(data_right), np.array(targets_right), depth + 1, create_leaf = True)
        return node


    def _find_best_feature_split(self, data, targets):
        min_criterion = 123456
        best_split, best_feature_index = None, None
        for feature_index, feature in enumerate(data.T):
            splits = self._get_split_values(feature)
            for split in splits:
                criterion_value = self._criterion_value(data, targets, feature_index, split)
                if criterion_value < min_criterion:
                    min_criterion = criterion_value
                    best_feature_index = feature_index
                    best_split = split
        return best_feature_index, best_split, min_criterion

    def _criterion_value(self, data, targets, feature_index, split):
        targets_l = [t for d, t in zip(data, targets) if d[feature_index] <= split]
        targets_r = [t for d, t in zip(data, targets) if d[feature_index] > split]
        return self.loss_fn[args.criterion](targets_l) + self.loss_fn[args.criterion](targets_r)



    def _get_split_values(self, feature_values):
        splits = []
        feature_values = sorted(feature_values)
        for i, _ in enumerate(feature_values):
            if i == 0:
                continue
            splits.append((feature_values[i - 1] + feature_values[i]) / 2)
        return splits


    def _count_pk(self, targets):
        number_of_targets = len(targets)
        p_k = []
        diff = []
        for t in targets:
            if t not in diff:
                diff.append(t)
                p_k.append(sum([1 if t2 == t else 0 for t2 in targets]) / number_of_targets)
        return p_k

    def _gini(self, targets):
        p_k = self._count_pk(targets)
        return len(targets) * sum([p * (1 - p) for p in p_k])

    def _entropy(self, targets):
        p_k = self._count_pk(targets)
        return -len(targets) * sum([p * math.log(p)  for p in p_k])

    
    def _find_most_frequent_item(self, targets):
        if len(targets) == 0:
            raise Exception("My expception")
        targets = sorted(targets)
        return_value = targets[0]
        max_occurence = 1
        occurence = 0
        for i, _ in enumerate(targets):
            if i == 0:
                continue
            if targets[i] == targets[i - 1]:
                occurence += 1
            else:
                occurence = 1
            if occurence > max_occurence:
                max_occurence = occurence
                return_value = targets[i]
        return return_value
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
    parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
    parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
    parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot progress")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_size", default=42, type=int, help="Test set size")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)





    # Use the digits dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)



    tree = DecisionTree(args)
    tree.build_tree(train_data, train_target)

    
    train_predictions = tree.predict(train_data)
    test_predictions = tree.predict(test_data)



    train_accuracy = sum([1 if t == p else 0 for t, p in zip(train_target, train_predictions)]) / len(train_target)
    test_accuracy = sum([1 if t == p else 0 for t, p in zip(test_target, test_predictions)]) / len(test_target)

    # TODO: Create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    # smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest feature values
    #   of the instances corresponding to the given node (i.e., for three instances
    #   with values 1, 7, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be at most `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    print("Train acc: {:.1f}%".format(100 * train_accuracy))
    print("Test acc: {:.1f}%".format(100 * test_accuracy))
