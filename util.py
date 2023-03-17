import numpy as np


def entropy(class_y):
    """
    This method calculates entropy
    Args:
        class_y: list of class labels
    Returns:
        entropy: entropy value

    Example: entropy for [0,0,0,1,1,1] should be 1.
    """
    # entropy = ∑(-p_i * log(p_i)) , where p_i donates ith index of p and
    # where p donates ratio of every unique element.

    # Unique values of class_y
    unique_values = np.unique(class_y)

    # Ratio of every unique element. It will be used for calculating their entropy -> p_i
    p = [class_y[class_y == unique].shape[0] / class_y.shape[0] for unique in unique_values]

    # Base 2 logarithm of every element in p
    log_p = np.log2(p)
    # Initially, entropy is zero.
    entropy_ = 0

    # Sum -p_i * log(p_i)
    for i in range(len(p)):
        entropy_ += -1 * p[i] * log_p[i]

    return entropy_


def information_gain(previous_y, current_y):
    """
    This method calculates information gain. In this method, use the entropy function you filled
    Args:
        y_before: the distribution of target values before splitting
        y_splitted: the distribution of target values after splitting

    Returns:
        information_gain

    Example: if y_before = [0,0,0,1,1,1] and y_splitted = [[0,0],[0,1,1,1]], information_gain = 0.4691

    """
    # Information gain equals to E(parent) - ∑ w_i * E(child_i)
    # Where w donates the ratio of each subset to the root set after the splitting.

    # Entropy of the parent(previous)
    entropy_parent = entropy(previous_y)

    # Entropies of the sub nodes
    left_entropy = entropy(current_y[0])
    right_entropy = entropy(current_y[1])
    ratio_left = current_y[0].shape[0] / previous_y.shape[0]
    ratio_right = current_y[1].shape[0] / previous_y.shape[0]

    # Info gain
    info_gain = entropy_parent - ratio_left*left_entropy - ratio_right*right_entropy

    return info_gain


def split_node(X, y, split_feature, split_value):
    """
    This method implements binary split to your X and y.
    Args:
        X: dataset without target value
        y: target labels
        split_feature: column index of the feature to split on
        split_value: value that is used to split X and y into two parts

    Returns:
        X_left: X values for left subtree
        X_right: X values for right subtree
        y_left: y values of X values in the left subtree
        y_right: y values of X values in the right subtree

    Notes:  Implement binary split.
            You can use mean for split_value or you can try different split_value values for better Random Forest results
            Assume you are only dealing with numerical features. You can ignore the case there are categorical features.
            Example:
                Divide X into two list.
                X_left: where values are <= split_value.
                X_right: where values are > split_value.
    """

    if split_feature == None:
        split_feature = 0

    X_left = []
    X_right = []

    y_left = []
    y_right = []
    # Iterating part
    for i in range(len(X)):
        # If the value at split feature is less or equal than split value add it to the left
        value = X[i][split_feature]

        if value <= split_value:
            X_left.append(X[i])
            y_left.append(y[i])

        # If the value at split feature is greater than split value add it to the right
        elif value > split_value:
            X_right.append(X[i])
            y_right.append(y[i])

        # If something else is about to happen, raise invalid value error.
        else:
            raise ValueError(f"Invalid value at {i}th index, values = {X[i],y[i]}")

    return np.array(X_left), np.array(X_right), np.array(y_left), np.array(y_right)


def confusion_matrix_(y_predicted, y):
    """
    Args:
        y_predicted: predicted number of features
        y: your true labels

    Returns:
        confusion_matrix: with shape (2, 2)

    """
    confusion_matrix = np.zeros((2, 2))
    # Iteration
    for i in range(len(y)):
        # If prediction is correct
        if y[i] == y_predicted[i]:
            # If actual value is 1
            if y[i]:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if y[i]:
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[1][0] += 1

    return confusion_matrix


def eval_metrics(conf_matrix):
    """
    Args:
        conf_matrix: Use confusion matrix you calculated

    Returns:
        accuracy, recall, precision, f1_score

    """
    accuracy, recall, precision, f1_score = 0, 0, 0, 0

    # a->TP, b->FN, c->FP, d->TN
    a, b, c, d = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]

    # accuracy = (TP+TN) / (TP+TN+FP+FN)
    accuracy = (a + d) / (a+b+c+d)

    # precision = TP / (TP + FP)
    precision = a / (a+c)

    # recall = TP / (TP + FN)
    recall = a / (a+b)

    # F-measure = 2*TP / (2*TP  + FN + FP)
    f1_score = 2*a / (2*a+b+c)

    return accuracy, recall, precision, f1_score
