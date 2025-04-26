import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0

    # Calculate the means of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate the Pearson correlation according to the formula
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    if denominator != 0:
        r = numerator / denominator

    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    correlations = {}  # Dictionary to store correlation values for each feature

    X_numeric = X.select_dtypes(include=np.number)  # Remove non-numeric features

    # Calculate the Pearson correlation for each feature
    for feature in X_numeric:
        correlations[feature] = pearson_correlation(X_numeric[feature].values, y)

    # Sort the features by their absolute correlation value in descending order
    sorted_features = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)

    # Select the top n_features features
    best_features = [feature for feature, _ in sorted_features[:n_features]]

    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, X):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Input vectors.
        """
        return 1 / (1 + np.exp(-np.dot(X, self.theta)))

    def compute_cost(self, X, y):
        """
        Compute the cost function.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors.
        y : array-like, shape = [n_examples]
          Target values.

        """
        h = self.sigmoid(X)
        numerator = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
        return numerator / len(y)

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # Add bias term to X
        X = np.insert(X, 0, 1, axis=1)
        # set random seed
        np.random.seed(self.random_state)
        # Initialize weights randomly
        self.theta = np.random.rand(X.shape[1])

        for i in range(self.n_iter):
            h = self.sigmoid(X)
            # Compute the gradient
            gradient = np.dot(X.T, (h - y))
            # Update the weights
            self.theta -= self.eta * gradient

            cost = self.compute_cost(X, y)
            # Store the cost and weights history
            self.Js.append(cost)
            self.thetas.append(self.theta.copy())

            # Check for convergence
            if i > 0 and abs(self.Js[-2] - cost) < self.eps:
                break

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None

        # Add bias term to X
        X = np.insert(X, 0, 1, axis=1)

        h = self.sigmoid(X)

        # Convert probabilities to class labels
        preds = np.where(h > 0.5, 1, 0)
        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Create folds
    fold_size = X.shape[0] // folds
    accuracies = []

    # train the model on each fold
    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size

        x_test = X[start:end]
        y_test = y[start:end]

        x_train = np.concatenate((X[:start], X[end:]))
        y_train = np.concatenate((y[:start], y[end:]))

        # Train the algo on the training data
        algo.fit(x_train, y_train)

        # Predict on the test data
        y_predict = algo.predict(x_test)

        # Calculate accuracy and append it to the list of accuracies
        accuracy = np.mean(y_predict == y_test)
        accuracies.append(accuracy)

    # Calculate the average accuracy over all folds
    cv_accuracy = np.mean(accuracies)
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    numerator = np.exp((-1) * np.square(data - mu)) / (2 * np.square(sigma))
    denominator = np.sqrt(2 * np.pi * np.square(sigma))
    p = numerator / denominator
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """

        self.weights = np.ones(self.k) / self.k
        self.mus = data[np.random.choice(data.shape[0], self.k, replace=False)].reshape(self.k)
        self.sigmas = np.random.random_integers(self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        curr_responsibilities = self.weights * norm_pdf(data, self.mus, self.sigmas)
        denominator = np.sum(curr_responsibilities, axis=1, keepdims=True)
        self.responsibilities = curr_responsibilities / denominator

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.sum(self.responsibilities, axis=0) / len(data)
        self.mus = np.sum(self.responsibilities * data.reshape(-1, 1), axis=0) / np.sum(self.responsibilities, axis=0)
        self.sigmas = np.sqrt(
            np.sum(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0) / np.sum(
                self.responsibilities, axis=0))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """

        self.init_params(data)
        cost = self.compute_cost(data)
        self.costs.append(cost)

        for i in range(self.n_iter):

            self.expectation(data)
            self.maximization(data)

            cost = self.compute_cost(data)
            self.costs.append(cost)

            if abs(self.costs[-2] - cost) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

    def compute_cost(self, data):
        weighted_pdfs = np.sum(self.weights * norm_pdf(data, self.mus, self.sigmas))
        cost = np.sum(-np.log(weighted_pdfs))
        return cost


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = np.sum(weights * norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)

    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.priors = None
        self.classes = None
        self.distributions = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """

        self.classes = np.unique(y)
        self.priors = {cls: len(y[y == cls]) / len(y) for cls in self.classes}
        self.distributions = {cls: [EM(self.k) for _ in range(X.shape[1])] for cls in self.classes}

        for cls in self.classes:
            for feature in range(X.shape[1]):
                self.distributions[cls][feature].fit(X[y == cls][:, feature].reshape(-1, 1))

    def compute_likelihood(self, X, cls):
        likelihood = np.ones(X.shape[0])
        for feature in range(X.shape[1]):
            weights, mus, sigmas = self.distributions[cls][feature].get_dist_params()
            gmm = gmm_pdf(X[:, feature], weights, mus, sigmas)
            likelihood *= gmm
        return likelihood

    def compute_posterior(self, X, cls):
        prior = self.priors[cls]
        likelihood = self.compute_likelihood(X, cls)
        posterior = prior * likelihood
        return posterior

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            label_posteriors = []
            for label in self.priors.keys():
                posterior = self.compute_posterior(instance.reshape(1, -1), label)
                label_posteriors.append((posterior, label))
            best_label = max(label_posteriors, key=lambda t: t[0])[1]
            preds.append(best_label)

        return np.array(preds).reshape(-1, 1)


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Logistic Regression with best params
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_y_train_predict = lor.predict(x_train)
    lor_train_acc = np.mean(lor_y_train_predict == y_train)
    lor_y_test_predict = lor.predict(x_test)
    lor_test_acc = np.mean(lor_y_test_predict == y_test)

    # Naive Bayes with best params
    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    bayes_y_train_predict = bayes.predict(x_train)
    bayes_train_acc = np.mean(bayes_y_train_predict == np.array(y_train).reshape(-1, 1))
    bayes_y_test_predict = bayes.predict(x_test)
    bayes_test_acc = np.mean(bayes_y_test_predict == np.array(y_test).reshape(-1, 1))

    # Plot decision boundaries for logistic regression and Naive Bayes
    for model, title in [(lor, "Logistic Regression"), (bayes, "Naive Bayes")]:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_decision_regions(x_train, y_train, classifier=model, title=f"{title} - Training Set")
        plt.subplot(1, 2, 2)
        plot_decision_regions(x_test, y_test, classifier=model, title=f"{title} - Test Set")
        plt.show()

    # Plot cost vs iteration for logistic regression
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lor.Js)), lor.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Logistic Regression - Cost vs Iteration')
    plt.show()

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None

    # Parameters for dataset_a (Naive Bayes performs better)
    mean1_class1_a = [0, 0, 0]
    cov1_class1_a = np.diag([2, 2, 2])
    mean2_class1_a = [15, 15, 15]
    cov2_class1_a = np.diag([2, 2, 2])
    mean1_class2_a = [5, 5, 5]
    cov1_class2_a = np.diag([2, 2, 2])
    mean2_class2_a = [20, 20, 20]
    cov2_class2_a = np.diag([2, 2, 2])

    np.random.seed(0)
    X_class1_a = np.vstack([
        np.random.multivariate_normal(mean1_class1_a, cov1_class1_a, 100),
        np.random.multivariate_normal(mean2_class1_a, cov2_class1_a, 100)
    ])
    y_class1_a = np.zeros(X_class1_a.shape[0])

    X_class2_a = np.vstack([
        np.random.multivariate_normal(mean1_class2_a, cov1_class2_a, 100),
        np.random.multivariate_normal(mean2_class2_a, cov2_class2_a, 100)
    ])
    y_class2_a = np.ones(X_class2_a.shape[0])

    dataset_a_features = np.vstack([X_class1_a, X_class2_a])
    dataset_a_labels = np.hstack([y_class1_a, y_class2_a])

    # Parameters for dataset_b (Logistic Regression performs better)
    mean_class1_b = [1, 1, 1]
    cov_class1_b = np.diag([0.5, 0.5, 0.5])
    mean_class2_b = [2, 2, 2]
    cov_class2_b = np.diag([0.5, 0.5, 0.5])

    X_class1_b = np.random.multivariate_normal(mean_class1_b, cov_class1_b, 200)
    y_class1_b = np.zeros(X_class1_b.shape[0])

    X_class2_b = np.random.multivariate_normal(mean_class2_b, cov_class2_b, 200)
    y_class2_b = np.ones(X_class2_b.shape[0])

    dataset_b_features = np.vstack([X_class1_b, X_class2_b])
    dataset_b_labels = np.hstack([y_class1_b, y_class2_b])

    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
