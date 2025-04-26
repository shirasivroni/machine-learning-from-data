# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    # Calculate the min, max and mean of X and y.
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_mean = np.mean(X, axis=0)

    y_max = np.max(y, axis=0)
    y_min = np.min(y, axis=0)
    y_mean = np.mean(y, axis=0)

    # Mean normalization of X and y
    X = (X - X_mean) / (X_max - X_min)
    y = (y - y_mean) / (y_max - y_min)

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    # Create a column of ones with the same length as X
    ones = np.ones((X.shape[0], 1))

    # Combine the ones and X into one array
    X = np.column_stack((ones, X))

    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    J = 0  # We use J for the cost.

    m = X.shape[0]  # Number of training instances
    h = np.dot(X, theta)
    error = h - y
    squared_error = error ** 2
    J = (1 / (2 * m)) * np.sum(squared_error)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = X.shape[0]  # Number of training instances

    # Update theta each iteration to find the lowest loss value.
    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    # (X^T * X)
    xtx = np.dot(X.T, X)

    # Inverse of (X^T * X)
    xtx_inverse = np.linalg.inv(xtx)

    # Pseudoinverse of X
    x_pinv = np.dot(xtx_inverse, X.T)

    # Compute the optimal theta
    pinv_theta = np.dot(x_pinv, y)

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration

    m = X.shape[0]  # Number of training instances

    # Update theta each iteration to find the lowest loss value.
    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        theta = theta - (alpha / m) * np.dot(X.T, error)
        J_history.append(compute_cost(X, y, theta))

        # Check if the improvement of the loss value is smaller than 1e-8.
        # If so, stop.
        if i > 0 and (J_history[i - 1] - J_history[i]) < 1e-8:
            break

    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    random_theta = np.random.random(X_train.shape[1])  # get random theta
    for alpha in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, random_theta, alpha, iterations)
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    # Loop until 5 features seleted
    while (len(selected_features) < 5):
        best_value_loss = float('inf')  # start with the worst loss
        best_feature = None

        # Iterate through all features that are not yet selected 
        for i in range(X_train.shape[1]):
            if i not in selected_features:
                features_to_test = selected_features + [i]  # new list with the new feature
                X_train_curr = X_train[:, features_to_test]
                X_val_curr = X_val[:, features_to_test]

                # Add bias 
                apply_bias_trick(X_train_curr)
                apply_bias_trick(X_val_curr)
                random_theta = np.random.random(X_train_curr.shape[1])  # get random theta
                theta, _ = efficient_gradient_descent(X_train_curr, y_train, random_theta, best_alpha, iterations)
                curr_value_loss = compute_cost(X_val_curr, y_val, theta)

                # Check if there was an improvment 
                if curr_value_loss < best_value_loss:
                    best_value_loss = curr_value_loss
                    best_feature = i

        # Add the best feature of this iteration to the list
        selected_features.append(best_feature)

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    new_cols = {}

    # Iterate over each column
    for i, feature1 in enumerate(df_poly.columns):
        for feature2 in df_poly.columns[i:]:
            if feature1 == feature2:
                feature = feature1 + '^2'
            else:
                feature = feature1 + '*' + feature2
            new_cols[feature] = df_poly[feature1] * df_poly[feature2]

    # Convert the dict to DataFrame
    new_cols_df = pd.DataFrame(new_cols)

    # Concat the original df with the new df
    df_poly = pd.concat([df_poly, new_cols_df], axis=1)

    return df_poly
