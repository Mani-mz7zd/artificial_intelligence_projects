# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    distance = 0
    l=len(x1)
    for i in range(l):
        distance += (x1[i] - x2[i])**2
    return distance ** 1/2
    #raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    distance = 0
    l=len(x1)
    for i in range(l):
        distance += abs(x1[i] - x2[i])
    return distance
    #raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        return 1

    return x

def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    sigmoid = 1.0 / (1.0 + np.exp(-x))
    if derivative:
        # derivative of sigmoid function is sigmoid*(1-sigmoid)
        return sigmoid * (1.0 - sigmoid)

    return sigmoid

def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        # derivative of tanh(x) is (1-(tanh(x))^2)
        return 1 - np.tanh(x)**2

    return np.tanh(x)

def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        # return 1 if x > 0 or 0 if x < 0
        return np.where(x >= 0, 1, 0)
    
    # return x if x > 0 or 0 if x < 0
    return np.where(x >= 0, x, 0)


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        s = softmax(x)
        return s * (1 - s)


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    m = y.shape[0]
    p = np.clip(p, 1e-15, 1 - 1e-15)

    return (-1.0/m) * np.sum(y * np.log(p))

def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    encoded_list = np.zeros((y.size, y.max()+1), dtype=int)

    #replacing 0 with a 1 at the index of the original array
    encoded_list[np.arange(y.size),y] = 1 
    # print(encoded_list)
    return encoded_list
