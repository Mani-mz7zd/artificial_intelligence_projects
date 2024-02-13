# Problem 1

## Report on Part-of-Speech Tagging Implementation

### Problem Formulation

1. Posterior Probability Calculation

Description:
The problem involves calculating the log of the posterior probability of a given sentence with a provided part-of-speech labeling using two models: Simple Bayes and Hidden Markov Model (HMM).

Formulation:
Utilized Bayes' rule to compute the posterior probability based on emission and transition probabilities. Implemented separate log probability calculations for the Simple and HMM models.

2. Training the Model

Description:
The task is to train the part-of-speech tagging model using input data, collecting word frequencies, and calculating prior word and part-of-speech probabilities.

Formulation:
Implemented a training method that iterates through the input data, updating word frequencies and counts. The training process involves capturing prior probabilities and previous part-of-speech transitions.

3. Part-of-Speech Labeling

Description:
The problem is to label each word in a given sentence with its corresponding part-of-speech using two algorithms: Simple Bayes and HMM.

Formulation:
For the Simple Bayes algorithm, assign each word the part-of-speech with the highest probability independently. For the HMM algorithm, implemented the Viterbi algorithm to find the most probable sequence of part-of-speech labels.

### Program Functionality

1. Posterior Probability Calculation

How it Works:
For Simple Bayes, iterates through each word in the sentence, calculates emission and prior probabilities, and accumulates the log probability. For HMM, it incorporates transition probabilities between part-of-speech labels.

Challenges and Decisions:
Ensured the use of logarithms to avoid underflow issues with small probabilities. Design decision to handle unknown words with a fallback grammar rule.

2. Training the Model

How it Works:
Processes input data to update word frequencies and counts. Sorts word frequencies for efficiency in posterior probability calculations.


# Problem 2

## Report on Optical Character Recognition Implementation:

### Problem Formulation

1. OCR using Simplified Approach

Description:
The first problem involves implementing an Optical Character Recognition (OCR) system using a simplified approach. This approach relies on calculating emission probabilities based on the similarity of test and training letters. The goal is to recognize characters in an image.

2. OCR using HMM with Viterbi Algorithm

Description:
The second problem extends the OCR system using a Hidden Markov Model (HMM) with the Viterbi algorithm. This algorithm is employed to find the most likely sequence of hidden states (characters) given the observed sequence (image).

### Program Functionality

1. OCR using Simplified Approach

How It Works:
Reads training text data, parsing and formatting it. Calculates emission probabilities based on black and white scores of characters. Recognizes characters in test images by selecting the letter with the maximum emission probability for each test letter.

2. OCR using HMM with Viterbi Algorithm

### How It Works:
Utilizes the Viterbi algorithm in the context of an HMM. Calculates emission probabilities, initial probabilities, and transition probabilities. Recognizes characters in test images by finding the most likely sequence of hidden states using the Viterbi algorithm.

Problems, Assumptions, and Design Decisions

1. OCR using Simplified Approach

Issues:
The emission probabilities are based on simple ratios of black and white pixels, which may not capture more complex patterns in the data. The approach assumes that a simple weighted combination of black and white scores is sufficient for character recognition. 

Assumptions and Design Decisions:
The choice of weights in calculating emission probabilities is arbitrary and may need further tuning for optimal results. Assumes that the simplified approach provides a reasonable baseline for character recognition.

2. OCR using HMM with Viterbi Algorithm

Issues:
The HMM implementation involves multiple probabilities (emission, initial, and transition) that may need fine-tuning for different datasets. The Viterbi algorithm's parameters, such as penalty weights, may impact the results.

Assumptions and Design Decisions:
Assumes that the Viterbi algorithm's parameters are appropriate for the given OCR problem. The training text data is assumed to be representative of the language and writing style in the test images.


### Reference Links
- https://towardsdatascience.com/training-hidden-markov-models-831c1bdec27d
- https://towardsdatascience.com/hidden-markov-models-an-overview-98926404da0e
- https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e
- https://www.cs.cmu.edu/~guestrin/Class/10701-S07/Handouts/recitations/HMM-inference.pdf 