# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Manikanta Kodandapani Naidu: k11, Pothapragada Venkata SG Krishna Srikar: vpothapr, G Vivek Reddy: gvi
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y
        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        test_data = X # Copy the test data matrix
        tr_data = self._X  # Get the training data and labels from the model
        tr_label = self._y
        result = [] # List to store the predicted labels for each test sample
        dist = [] # List to store distances from the test sample to each training sample
   
        for test_sample in test_data:
            dist = []
            for tr_sample in tr_data: # Calculate the distance between the test sample and each training sample
                dist.append(self._distance(test_sample,tr_sample))
            sorted_labels  = [label for _, label in sorted(zip(dist, tr_label))]

            # Take the majority class among the k-nearest neighbors
            k_nearest_labels = sorted_labels[0:self.n_neighbors]
            pre_label = max(k_nearest_labels, key=k_nearest_labels.count)
            result.append(pre_label)
            
        return result
        #raise NotImplementedError('This function must be implemented by the student.')
