import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

# Written by : 212926091 & 211392329

chi_table = {
    1: {0.5: 0.45, 0.25: 1.32, 0.1: 2.71, 0.05: 3.84, 0.0001: 100000},
    2: {0.5: 1.39, 0.25: 2.77, 0.1: 4.60, 0.05: 5.99, 0.0001: 100000},
    3: {0.5: 2.37, 0.25: 4.11, 0.1: 6.25, 0.05: 7.82, 0.0001: 100000},
    4: {0.5: 3.36, 0.25: 5.38, 0.1: 7.78, 0.05: 9.49, 0.0001: 100000},
    5: {0.5: 4.35, 0.25: 6.63, 0.1: 9.24, 0.05: 11.07, 0.0001: 100000},
    6: {0.5: 5.35, 0.25: 7.84, 0.1: 10.64, 0.05: 12.59, 0.0001: 100000},
    7: {0.5: 6.35, 0.25: 9.04, 0.1: 12.01, 0.05: 14.07, 0.0001: 100000},
    8: {0.5: 7.34, 0.25: 10.22, 0.1: 13.36, 0.05: 15.51, 0.0001: 100000},
    9: {0.5: 8.34, 0.25: 11.39, 0.1: 14.68, 0.05: 16.92, 0.0001: 100000},
    10: {0.5: 9.34, 0.25: 12.55, 0.1: 15.99, 0.05: 18.31, 0.0001: 100000},
    11: {0.5: 10.34, 0.25: 13.7, 0.1: 17.27, 0.05: 19.68, 0.0001: 100000},
}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0

    # Handling empty data.
    if data.shape[0] == 0:
        return 0.0

    # The last column holds the labels.
    labels = data[:, -1]

    _, labels_count = np.unique(labels, return_counts=True)

    # Calculate the probabilities.
    probabilities = labels_count / labels_count.sum()

    # Calculate the gini impurity.
    gini = 1 - np.sum(probabilities ** 2)

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0

    # Handling empty data.
    if data.shape[0] == 0:
        return 0.0

    # The last column holds the labels.
    labels = data[:, -1]

    _, labels_count = np.unique(labels, return_counts=True)

    # Calculate the probabilities.
    probabilities = labels_count / labels_count.sum()

    # Calculate the entropy impurity.
    entropy = -np.sum(
        probabilities * np.log2(probabilities + np.finfo(float).eps)
    )  # Adding epsilon to avoid log(0).

    return entropy


class DecisionNode:

    def __init__(
            self,
            data,
            impurity_func,
            feature=-1,
            depth=0,
            chi=1,
            max_depth=1000,
            gain_ratio=False,
    ):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None

        # Handling empty data.
        if self.data.shape[0] == 0:
            return None

        # The last column holds the labels.
        labels = self.data[:, -1]

        # Find most frequent label.
        values, counts = np.unique(labels, return_counts=True)
        pred = values[np.argmax(counts)]

        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """

        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """

        # If node is a leaf or none, feature importance is 0.
        if self.feature is None or self.terminal:
            self.feature_importance = 0
            return

        # Calculate the impurity reduction.
        impurity_reduction, _ = self.goodness_of_split(self.feature)

        # Calculate the feature importance.
        ratio = len(self.data) / n_total_sample
        self.feature_importance = ratio * impurity_reduction

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}

        # Calculate the impurity before the split.
        if self.gain_ratio:
            total_impurity = calc_entropy(self.data)
        else:
            total_impurity = self.impurity_func(self.data)

        # Get the column of the data relevent to the feature.
        feature_column = self.data[:, feature]
        values, _ = np.unique(feature_column, return_counts=True)

        # Create groups for data split by feature values.
        groups = {val: self.data[feature_column == val] for val in values}

        # Calculate imputiry after the split.
        weighted_impurity = 0
        split_information = 0
        for val in values:
            subset_impurity = self.impurity_func(groups[val])
            subset_ratio = len(groups[val]) / len(self.data)
            weighted_impurity += subset_impurity * subset_ratio

            # Calculate split_information for this subset
            if subset_ratio > 0:  # Avoid log(0)
                split_information -= subset_ratio * np.log2(subset_ratio)

        # Goodness of split.
        goodness = total_impurity - weighted_impurity

        # If gain_ratio is set to True, calculate and return gain ratio.
        if self.gain_ratio:
            if split_information == 0:  # Avoid division by zero
                return 0, groups

            information_gain = goodness  # Just for the formula

            # Calculate gain ratio
            gain_ratio = information_gain / split_information

            goodness = gain_ratio  # Just for the return

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """

        # Stop spliting if the node reaches the maximum depth.
        if self.depth == self.max_depth:
            self.terminal = True
            return

        features = self.data.shape[1] - 1  # The last column holdes the labels.
        best_feature = None
        best_impurity_reduction = -np.inf
        best_groups = {}

        # Find the best feature
        for feature in range(features):
            impurity_reduction, groups = self.goodness_of_split(feature)
            if impurity_reduction > best_impurity_reduction:
                best_feature = feature
                best_impurity_reduction = impurity_reduction
                best_groups = groups

        # If there's no best feature or there is one child or less, set as leaf.
        if best_feature is None or len(best_groups) < 1 or best_impurity_reduction <= 0:
            self.terminal = True
            return

        self.feature = best_feature

        # Support pruning according to self.chi. If chi is 1 no pruning is done.
        if self.chi != 1:
            chi_square, chi_from_table = calculate_chi_square(self)
            if chi_square < chi_from_table:
                self.terminal = True
                return

        # Create the children.
        for value, subset in best_groups.items():
            child_node = DecisionNode(
                data=subset,
                impurity_func=self.impurity_func,
                feature=best_feature,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio,
            )
            self.add_child(child_node, value)


def calculate_chi_square(node):
    """
    Calculate the chi square statistic and retrieve the critical chi square value from the chi table.

    Input:
    - node: the node to calculate the chi on.

    Returns:
    - chi_square: the computed chi square statistic.
    - chi_from_table: the critical chi square value from the chi table for the node's chi value and degrees of freedom.
    """

    # Extract labels from the last column of the node's data.
    labels = node.data[:, -1]
    # Get unique labels and their counts.
    unique_labels, laels_counts = np.unique(labels, return_counts=True)

    # Extract the feature column that we're analyzing from the node's data.
    feature_column = node.data[:, node.feature]
    # Get unique values of the feature and their counts.
    unique_values, values_counts = np.unique(feature_column, return_counts=True)

    total_samples = len(labels)

    # Calculate expected counts for each combination of feature values and labels.
    expected_counts = np.zeros((len(unique_values), len(unique_labels)))
    for i, v_count in enumerate(values_counts):
        for j, l_count in enumerate(laels_counts):
            expected_counts[i, j] = (v_count * l_count) / total_samples

    # Compute observed counts for each combination of feature values and labels.
    observed_counts = np.zeros_like(expected_counts)
    for i, value in enumerate(unique_values):
        for j, label in enumerate(unique_labels):
            observed_counts[i, j] = np.sum(
                (feature_column == value) & (labels == label)
            )

    # Calculate the chi square statistic.
    chi_square = np.sum(
        (observed_counts - expected_counts) ** 2 / expected_counts,
        where=(expected_counts > 0),
    )
    # Calculate degrees of freedom for the chi-square test.
    degree_of_freedom = (len(unique_values) - 1) * (len(unique_labels) - 1)
    # Retrieve the critical chi square value from the table.
    chi_from_table = chi_table[degree_of_freedom][node.chi]

    return chi_square, chi_from_table


class DecisionTree:
    def __init__(
            self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False
    ):
        self.data = data  # the relevant data for the tree
        self.impurity_func = (
            impurity_func  # the impurity function to be used in the tree
        )
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None

        # Create the root.
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            max_depth=self.max_depth,
            chi=self.chi,
            gain_ratio=self.gain_ratio,
        )

        # Create a queue to store the nodes.
        queue = [self.root]
        while queue:

            current_node = queue.pop(0)

            # If the tree reaches to the max_depth or if all the labels are the same, set as leaf.
            if (
                    current_node.depth >= self.max_depth
                    or np.unique(current_node.data[:, -1]).size == 1
            ):
                current_node.terminal = True
                continue

            # Split the current node.
            current_node.split()

            # Add the children
            if current_node.children:
                queue.extend(current_node.children)
                current_node.calc_feature_importance(self.data.shape[0])

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None

        node = self.root
        while not node.terminal:
            feature_value = instance[node.feature]

            # If the feature value is not in the children, return the prediction of the current node.
            if feature_value not in node.children_values:
                return node.pred

            # Else, go to the child node with the feature value.
            else:
                node = node.children[node.children_values.index(feature_value)]

        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0

        correct_prediction = 0

        # Iterate over the instances in the given dataset.
        for instance in dataset:

            # Get the prediction of the current instance.
            prediction = self.predict(instance)

            # Check if the prediction is correct.
            if prediction == instance[-1]:
                correct_prediction += 1

        # Calculate the accuracy of the decision tree on the given dataset.
        accuracy = (correct_prediction / len(dataset)) * 100

        return accuracy

    def depth(self):
        if self.root.terminal:
            return self.root.depth
        return calculate_depth(self.root)


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Initialize the decision tree.
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            max_depth=max_depth,
            gain_ratio=True,
        )
        tree.build_tree()

        # Calculate the accuracy on the training data
        train_accuracy = tree.calc_accuracy(X_train)
        training.append(train_accuracy)

        # Calculate the accuracy on the validation data
        validation_accuracy = tree.calc_accuracy(X_validation)
        validation.append(validation_accuracy)

    return training, validation


def calculate_depth(node):
    """
    Calculate the depth of the tree starting from the root.

    Input:
    - node : a node from the tree.

    Output:
    - max_depth : the depth of the tree.
    """

    # If the node is a leaft, return it's depth.
    if not node.children or node.terminal:
        return node.depth

    # If the node has children, calculate the depth recursively and find the maximum depth among the children.
    max_depth = max(calculate_depth(child) for child in node.children)

    return max_depth


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for chi in chi_values:
        # Initialize the decision tree.
        tree = DecisionTree(
            data=X_train, impurity_func=calc_entropy, chi=chi, gain_ratio=True
        )
        tree.build_tree()

        # Calculate the accuracy on the training data
        train_accuracy = tree.calc_accuracy(X_train)
        chi_training_acc.append(train_accuracy)

        # Calculate the accuracy on the validation data
        validation_accuracy = tree.calc_accuracy(X_test)
        chi_validation_acc.append(validation_accuracy)

        # Calculate the depth of the tree.
        depth.append(tree.depth())

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """

    # Base case.
    if node is None:
        return 0

    # Start count from 1 for the current node.
    n_nodes = 1

    # Recursively count all nodes.
    for child in node.children:
        n_nodes += count_nodes(child)

    return n_nodes
