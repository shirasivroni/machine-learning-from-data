import numpy as np


class conditional_independence():

    def __init__(self):

        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        # X and Y are not independent, meaning P(X, Y) != P(X) * P(Y)
        self.X_Y = {
            (0, 0): 0.1,  # != 0.09
            (0, 1): 0.2,  # != 0.21
            (1, 0): 0.2,  # != 0.21
            (1, 1): 0.5  # != 0.49
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.1,  # sum to 0.3, P(x=0) = 0.3
            (0, 1): 0.2,
            (1, 0): 0.4,  # sum to 0.7, P(x=1) = 0.7
            (1, 1): 0.3
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.1,  # sum to 0.3, P(y=0) = 0.3
            (0, 1): 0.2,
            (1, 0): 0.4,  # sum to 0.7, P(y=1) = 0.7
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        # X and Y are conditionally independent given C, meaning P(X, Y, C) = P(X, Y | C) * P(C)
        self.X_Y_C = {
            (0, 0, 0): 0.02,  # 0.1 * 0.1 / 0.5
            (0, 0, 1): 0.08,  # 0.2 * 0.2 / 0.5
            (0, 1, 0): 0.08,  # 0.1 * 0.4 / 0.5
            (0, 1, 1): 0.12,  # 0.2 * 0.3 / 0.5
            (1, 0, 0): 0.08,  # 0.4 * 0.1 / 0.5
            (1, 0, 1): 0.12,  # 0.3 * 0.2 / 0.5
            (1, 1, 0): 0.32,  # 0.4 * 0.4 / 0.5
            (1, 1, 1): 0.18,  # 0.3 * 0.3 / 0.5
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y

        # Checking if X and Y are independent, meaning P(X, Y) == P(X) * P(Y).
        for x, y in X_Y.keys():
            if not np.isclose(X_Y[(x, y)], X[x] * Y[y]):
                return True
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        for x in X.keys():
            for y in Y.keys():
                for c in C.keys():
                    x_given_c = X_C[(x, c)] / C[c] if C[c] > 0 else 0
                    y_given_c = Y_C[(y, c)] / C[c] if C[c] > 0 else 0
                    xy_given_c = X_Y_C[(x, y, c)] / C[c] if C[c] > 0 else 0
                    if not np.isclose(xy_given_c, y_given_c * x_given_c):
                        return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    neg_rate = (-1) * rate
    numerator = (rate ** k) * np.exp(neg_rate)
    denominator = np.math.factorial(k)
    log_p = np.log(numerator / denominator)
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    # Initialize an array to store the log-likelihoods for each rate.
    likelihoods = np.zeros(len(rates))

    # Loop through each rate in the list of rates.
    for i, rate in enumerate(rates):
        # Compute the log PMF for each sample at the current rate and sum.
        log_likelihood = sum(poisson_log_pmf(k, rate) for k in samples)
        likelihoods[i] = log_likelihood

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0

    # Calculate the log-likelihoods for all rates
    likelihoods = get_poisson_log_likelihoods(samples, rates)

    # Find the index of the maximum log likelihood
    max_index = np.argmax(likelihoods)

    rate = rates[max_index]

    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """

    # Compute the sample mean.
    mean = np.mean(samples)

    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    e_power = (-1) * (((x - mean) ** 2) / (2 * (std ** 2)))
    numerator = np.exp(e_power)
    denominator = np.sqrt(2 * np.pi * (std ** 2))
    p = numerator / denominator
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """

        self.dataset = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(self.class_data, axis=0)
        self.std = np.std(self.class_data, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = normal_pdf(x[:-1], self.mean, self.std).prod()
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1
        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    count = 0
    for x in test_set:
        if map_classifier.predict(x) == x[-1]:
            count += 1
    acc = count / len(test_set)
    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = mean.shape[0]
    left = (2 * np.pi) ** (-d / 2) * np.linalg.det(cov) ** (-0.5)
    diff = x - mean
    exp = -0.5 * np.dot(diff.T, np.dot(np.linalg.inv(cov), diff))
    right = np.exp(exp)
    pdf = left * right
    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        self.dataset = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(self.class_data, axis=0)
        self.cov = np.cov(self.class_data.T)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x[:-1], self.mean, self.cov)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            pred = 0
        else:
            pred = 1
        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            pred = 0
        else:
            pred = 1
        return pred


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_data = dataset[dataset[:, -1] == class_value]

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_data) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1.0
        n_i = len(self.class_data)

        # Loop over each feature in the instance
        for feature in range(self.class_data.shape[1] - 1):
            featureValues = self.class_data[:, feature]
            if x[feature] not in featureValues:
                likelihood *= EPSILLON
            else:
                v_j = len(set(featureValues))
                n_i_j = np.sum(featureValues == x[feature])
                # Laplace estimation
                likelihood *= (n_i_j + 1) / (n_i + v_j)

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        count = 0
        for x in test_set:
            if self.predict(x[:-1]) == x[-1]:
                count += 1
        acc = count / len(test_set)
        return acc
