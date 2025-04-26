import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []

    # Select k unique indices
    num_pixels = X.shape[0]
    indices = np.random.choice(num_pixels, k, replace=False)
    centroids = X[indices, :]

    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)

    # num_pixels = X.shape[0]

    # distances = np.zeros((k, num_pixels))
    # # Calculate the Minkowski distance for each centroid and each pixel
    # for i in range(k):
    #     distances[i, :] = np.sum(np.abs(X - centroids[i])**p, axis=1)**(1/p)

    differences = X[np.newaxis, :, :] - centroids[:, np.newaxis, :]
    distances = np.sum(np.abs(differences) ** p, axis=2) ** (1 / p)

    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)

    num_pixels = X.shape[0]
    for _ in range(max_iter):
        # Calculate distances between each point and each centroid
        distances = lp_distance(X, centroids, p)

        # Assign each point to the closest centroid
        classes = np.argmin(distances, axis=0)

        # Calculate new centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            points = X[classes == i]
            if len(points) > 0:
                new_centroids[i] = np.mean(points, axis=0)
            else:
                new_centroids[i] = X[np.random.choice(num_pixels)]

        # Check for convergence 
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None

    num_pixels = X.shape[0]

    # Choose a centroid uniformly at random among the data points.
    first_centroid = np.random.choice(num_pixels)
    centroids = [X[first_centroid]]

    # Choose the remaining centroids using a weighted probability
    for _ in range(1, k):
        distances = lp_distance(X, np.array(centroids), p)
        min_distances = np.min(distances, axis=0)
        squared_distances = min_distances ** 2
        total_distances = np.sum(squared_distances)
        probabilities = squared_distances / total_distances
        next_centroid = np.random.choice(num_pixels, p=probabilities)
        centroids.append(X[next_centroid])

    centroids = np.asarray(centroids).astype(np.float64)

    # Now that the initial centroids have been chosen, proceed using standard k-means clustering.
    for _ in range(max_iter):
        # Calculate distances between each point and each centroid
        distances = lp_distance(X, centroids, p)

        # Assign each point to the closest centroid
        classes = np.argmin(distances, axis=0)

        # Calculate new centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            points = X[classes == i]
            if len(points) > 0:
                new_centroids[i] = np.mean(points, axis=0)
            else:
                new_centroids[i] = X[np.random.choice(num_pixels)]

        # Check for convergence 
        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return centroids, classes
