# gvi-vpothapr-k11-a4

## Part 1: K-Nearest Neighbors (KNN) Classifier Implementation Report

## 1. Problem Formulation:
The task was to implement a K-Nearest Neighbors (KNN) classifier from scratch in Python. The goal was to create a flexible and functional machine learning model that allows users to specify the number of neighbors (n_neighbors), the weight function (weights), and the distance metric (metric). The implementation was expected to handle both Euclidean and Manhattan distance metrics.

## 2. Program Description:

- **KNearestNeighbors Class:** The KNearestNeighbors class is the core of the program, encapsulating the KNN classifier. It is initialized with parameters such as the number of neighbors, weight function, and distance metric. The fit method stores the training data and labels, while the predict method predicts class target values for test data based on the KNN algorithm.

- **Distance Metric Functions (in utils.py):** The utility file (utils.py) contains functions for computing Euclidean and Manhattan distances. These functions are used by the KNN classifier to calculate distances between data points.

## 3. Program Workflow:

- **Initialization:** The KNearestNeighbors class is instantiated with user-specified parameters.
- **Fit Method:**  The fit method is called with training data and labels, storing them in the object.
- **Predict Method:** For each test sample, distances to all training samples are calculated using the specified distance metric. The labels of training samples are sorted based on distances. The majority class among the k-nearest neighbors is determined, and the predicted label is appended to the result list. The process is repeated for all test samples.


## Part 2: Multilayer Perceptron Classification

## 1. Data Preparation and One-Hot Encoding
- **Data Split:** The dataset is divided into training and testing sets, ensuring a separate set of data for model evaluation.
- **One-Hot Encoding:** The categorical target values are converted into a binary matrix format, known as one-hot encoding. This is crucial for handling multiclass classification problems efficiently.

## 2. Weight Initialization
- **Random Weight Assignment:** Initial weights for the perceptrons in both the hidden and output layers are assigned randomly. The method takes into account the number of features in the input data and the number of neurons in the hidden layer.
- **Xavier Initialization:** For activation functions like sigmoid and tanh, Xavier initialization is applied to stabilize training by scaling weights appropriately.
- **He Initialization:** For the ReLU activation function, He initialization is used to address the vanishing gradient problem.

## 3. Bias Initialization
- **Random Bias Assignment:** Biases for both the hidden and output layers are initialized with random values. These biases introduce flexibility to the model and help in capturing patterns that might not be represented in the input features alone.

## 4. Forward Pass
- **Activation Function Application:** The training data is passed through the neural network's hidden layer using the specified activation function (defaulting to sigmoid if not specified). This creates an activation output for the hidden layer.
- **Output Layer Activation:** The activation output from the hidden layer is then passed to the output layer, applying the specified activation function (defaulting to softmax). The final output layer activation values are obtained.

## 5. Loss Computation
- **Cross-Entropy Loss:** The cross-entropy loss function is employed to compute the error between the predicted output and the true labels. This loss serves as the measure of how well the model is performing.

## 6. Backward Pass (Training)
- **Gradient Descent:** The model's weights and biases are updated in the backward pass using gradient descent. This involves calculating the gradients of the loss with respect to the weights and biases and adjusting them to minimize the loss.

## 7. Predict Function
- **Forward Pass for Prediction:** Similar to the training phase, the predict function applies the forward pass to the test dataset using the learned weights and biases.
- **Class Prediction:** The class with the highest probability in the output layer is considered as the predicted class for each test sample.
