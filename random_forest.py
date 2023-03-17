#Author: @mserdarnazli(turkchen)


import numpy

from decision_tree import DecisionTree
from util import confusion_matrix_, eval_metrics
import csv
import numpy as np
import ast


class RandomForest(object):
    num_trees = 0
    decision_trees = []
    # Includes bootstrapped datasets. List of list
    bootstrap_datasets = []
    # Corresponding class labels for above bootstraps_datasets
    bootstrap_labels = []

    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.decision_trees = [DecisionTree() for i in range(n_trees)]

    def create_bootstrap_dataset(self, XX, n):
        """
        In this method, create sample datasets with size n by sampling with replacement from XX.
        
        Args:
            XX: original dataset (includes target feature y)
            n: sampling size

        Returns:
            samples: sampled data (excluding target feature y)
            labels:

        """

        samples = []  # sampled dataset
        labels = []  # class labels for the sampled records

        # Converting the dataset to numpy array to work on it easily.
        XX = numpy.array(XX)
        indexes = np.random.randint(0, len(XX), n)

        for i in indexes:
            sample = XX[i, :-1]
            label =  XX[i, -1]
            samples.append(sample.tolist())
            labels.append(label.tolist())

        return (samples, labels)

    def bootstrapping(self, XX):
        """
        This method initializes the bootstrap datasets for each tree
        Args:
            XX
        """
        for i in range(self.n_trees):
            data_sample, data_label = self.create_bootstrap_dataset(XX, len(XX))
            self.bootstrap_datasets.append(data_sample)
            self.bootstrap_labels.append(data_label)

    def fitting(self):
        """
        This method train each decision tree (with number of n_trees) using the bootstraps datasets and labels
        """
        # Iteration over the trees
        for i in range(self.n_trees):
            # Printing which tree is in training
            print(f"tree_{i} is in training.")

            # Training the tree
            self.decision_trees[i].train(self.bootstrap_datasets[i], self.bootstrap_labels[i])

    def majority_voting(self, X):
        """
        This method predicts labels for X using all the decision tree we fit.
        Args:
            X: dataset

        Returns:
            y: predicted value for each data point (includes the prediction of each decision tree)

        Explanation:
            After finding the predicted labels by all the decision tree, use majority voting to choose one class label
            Example: if 3 decision tree find the class labels as [1, 1, 0],
                    the majority voting should find the predicted label as 1
                    because the number of 1's is bigger than the number of 0's
        """

        y = []  # Holds the majority voted values

        # Iteration over each record
        for record in X:
            # Taking the votes of all decision trees.
            votes = [self.decision_trees[i].classify(record, one=True) for i in range(self.n_trees)]

            # Finding the major vote
            major_vote = max(votes, key=votes.count)

            y.append(major_vote)

        # Termination
        return y


def read_dataset():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels

    # Loading data set
    print("...Reading Airline Passenger Satisfaction Dataset...")
    with open("example_dataset/airline_passenger_satisfaction.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                xline.append(ast.literal_eval(line[i]))

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])
    return X, y, XX


def main():
    """
    X: matrix with n rows and d columns where n is the number of rows and d is the number of features
    y: target features with length n
    XX: X + y
    """
    X, y, XX = read_dataset()

    #  You can change below
    forest_size = 1

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)
    randomForest.bootstrapping(XX)

    print("...Fit your forest (all your decision trees)...")
    randomForest.fitting()

    print("...Make Prediction...")
    y_predicted = randomForest.majority_voting(X)

    matr = confusion_matrix_(y_predicted, y)
    accuracy, recall, precision, f1_score = eval_metrics(matr)
    print(matr)
    print("accuracy: %.4f" % accuracy)
    print("recall: %.4f" % recall)
    print("precision: %.4f" % precision)
    print("f1_score: %.4f" % f1_score)


if __name__ == "__main__":
    main()
