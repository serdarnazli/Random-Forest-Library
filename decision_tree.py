#Author: @mserdarnazli(turkchen)



from util import entropy, information_gain, split_node
import numpy as np


class Node(object):
    def __init__(self, split_value=None, split_feature=None, left=None, right=None, is_leaf=None, value=None):
        # Variable names are self-explanatory
        self.split_value = split_value
        self.split_feature = split_feature
        self.left = left
        self.right = right

        # Leaf nodes will carry 0 or 1(class names actually)
        self.is_leaf = is_leaf
        self.value = value


class DecisionTree(object):
    def __init__(self, max_depth=5):
        self.tree = None            # you can use different data structure to build your tree
        self.min_samples_split = 2  # one of the conditions of termination of growing the tree.
        self.max_depth = max_depth  # Max depth, one of the conditions of termination of growing the tree.
        self.root = None            # Initially it is None.

    def train(self, X, y):
        """
        This method trains decision tree (trains = construct = build)
        Args:
            X: data excluded target feature
            y: target feature

        Returns:
            None

        """
        # Converting them to np.arrays to work easily on them.
        X = np.array(X)
        y = np.array(y)

        # Constructing the tree.
        self.root = self.construct_tree(X, y)

    def find_best_split_value(self, X, y, random_feature_indexes):
        """
        This method finds the best split value and split feature for a decision node.
        Args:
            X: dataset excluded target feature
            y: target feature
            random_feature_indexes: some feature indexes which is selected randomly.
        Returns:
            A dictionary which has keys as info_gain,split_value,split_feature
        """

        # To hold the best split information, we will use this.
        best_split_informations = {"info_gain": float("-inf"), "split_value": None, "split_feature": None, "left": None,
                                   "Right": None}

        # Iterating the possible feature indexes.
        for ftr_idx in random_feature_indexes:
            # Finding possible split values
            all_split_values = np.unique(X[:, ftr_idx])

            # Iterating every possible split value
            for split_value in all_split_values:

                # Splitting process
                X_left, X_right, y_left, y_right = split_node(X, y, split_value=split_value, split_feature=ftr_idx)
                if not (X_left.shape[0] > 0 and X_right.shape[0] > 0):
                    continue

                # Calculating the information gain
                info_gain = information_gain(y, [y_left, y_right])
                # If the calculated info gain is greater than the one we had, it is the new best choice.
                if info_gain > best_split_informations["info_gain"]:

                    # Setting this option as best.
                    best_split_informations["info_gain"] = info_gain
                    best_split_informations["split_value"] = split_value
                    best_split_informations["split_feature"] = ftr_idx

        # Termination
        return best_split_informations

    def construct_tree(self, X, y,depth=0):
        """
        Top-down recursive divide-and-conquer function(look at the 29th page of the slide # 3)
        Args:
            X: dataset excluded target feature
            y: target feature
            depth: to check in what depth the processes are going.
        Returns:
            Root node.

        """

        # How many random feature are we going to select.
        how_many_random_feature = 5

        # Randomly selecting feature indexes.
        random_feature_indexes = np.random.randint(0, X.shape[1], how_many_random_feature)
        # Taking the only unique ones.
        random_feature_indexes = np.unique(random_feature_indexes)

        # How many sample dataset includes
        number_of_samples = X.shape[0]

        # number_of_features = len(random_feature_indexes)

        # If there are not enough samples to split, termination of the recursive function.
        if number_of_samples <= self.min_samples_split:
            # What is the value that leaf hold. 1 or 0
            value_leaf_hold = max(list(y), key=list(y).count)
            return Node(is_leaf=True, value=value_leaf_hold)

        # If we are in the maximum depth, termination of the recursive function.
        if depth >= self.max_depth:
            value_leaf_hold = max(list(y),key=list(y).count)
            return Node(is_leaf=True, value=value_leaf_hold)

        # Finding the best split informations
        best_split_informations = self.find_best_split_value(X, y, random_feature_indexes)

        if best_split_informations["info_gain"] <= 0:
            value_leaf_hold = max(list(y),key=list(y).count)
            return Node(is_leaf=True, value=value_leaf_hold)

        # Best split value
        split_value = best_split_informations["split_value"]

        # Best split feature (as index)
        split_feature = best_split_informations["split_feature"]

        # Splitting the dataset from best split information
        X_left, X_right, y_left, y_right = split_node(X, y, split_value=split_value, split_feature=split_feature)

        # Converting to np.array data structure
        X_left, X_right, y_left, y_right = np.array(X_left), np.array(X_right), np.array(y_left), np.array(y_right)

        # Recursively constructing the left and right trees
        left_tree = self.construct_tree(X_left, y_left, depth+1)
        right_tree = self.construct_tree(X_right, y_right, depth+1)

        # Termination
        return Node(split_feature=split_feature, split_value=split_value, is_leaf=False, left=left_tree,
                    right=right_tree, value=None)

    def classify(self, record, one=False):
        """
        This method classifies the record using the tree you build and returns the predicted la
        Args:
            record: each data point

        Returns:
            predicted_label

        """

        # If there is only one record to classify:
        if one:
            return self.go_leaf(self.root, record)

        # Predicted labels are initially empty.
        predicted_label = []

        # Predicting every record
        for rec in record:
            predicted_label.append(self.go_leaf(self.root, rec))

        # Termination
        return predicted_label

    def go_leaf(self, start_node, record):
        """
        This recursive method starts from a node and goes 1 level deeper in each call. At the end
        it returns the predicted value that record has.
        Args:
            start_node: where to start
            record:     record to predict
        Returns:
            prediction
        """
        # If we are already at the leaf, termination of the recursive.
        if start_node.is_leaf:
            return start_node.value

        # The feature that the node making decisions
        feature = start_node.split_feature
        # The value that node splits from
        value = start_node.split_value

        # If the value at the feature of the record is less or equal than the value that node splits from, going left
        if record[feature] <= value:
            return self.go_leaf(start_node.left, record)

        # Else, going to right sub-node
        else:
            return self.go_leaf(start_node.right, record)
