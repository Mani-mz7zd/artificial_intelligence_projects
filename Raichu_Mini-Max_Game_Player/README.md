# Problem 1:
Report for Raichu Game

Abstract:
This report presents an analysis of the Raichu game implementation, including the code and the decision-making algorithm used to determine the best moves for a given player. The Raichu game is a two-player board game where the objective is to outmaneuver the opponent by making strategic moves. This report outlines the code's functionality, implementation details, challenges faced, and design decisions made.

1. Introduction:
The Raichu game is a two-player board game where one player controls white ('w') pieces and the other controls black ('b') pieces. The board is represented as a grid, and players take turns to move their pieces with the goal of capturing their opponent's pieces and reaching a winning position.

2. Problem Formulation:
The problem is to implement a game-playing program for Raichu that can evaluate board states and suggest optimal moves. The code aims to determine the best moves for a player given a specific board state.

3. Program Overview:
The code consists of functions and logic to represent the game board, pieces, player moves, and the decision-making algorithm. The board is represented as a two-dimensional grid, and players are designated as 'w' and 'b'.

The code performs the following functions:

- Implements the game board as a 2D grid.
- Provides functions to convert the board into strings for display.
- Determines the locations of pieces for each player.
- Evaluates the current state of the game board and assigns a score.
- Generates possible moves for each player, considering different types of pieces (Raichu, Pikachu, Pichu).
- Utilizes the minimax algorithm with alpha-beta pruning to find the best move for a player.



4. Implementation Details:
The code utilizes various functions to perform actions such as evaluating the current board state, generating valid moves, and determining the best move for a player. Functions like `generate_moves`, `Pikachu_moves`, and `Raichu_moves` are crucial for generating valid moves for each piece type.

5. Challenges and Problems Faced:
The code addresses challenges such as move validation, handling different piece types, and ensuring that the game rules are followed. These challenges were overcome by implementing functions to check for valid moves and by handling different scenarios in the game.

6. Assumptions and Simplifications:
The code assumes a simplified version of the Raichu game, considering only two players and a fixed-size board. The code also assumes that the input board is valid and conforms to the game rules.

7. Design Decisions:
Several design decisions were made to create an efficient game-playing program. The code uses a decision-making algorithm based on evaluating board states and potential moves. The design choices include piece values and move rules.

8. Results and Evaluation:
The code is designed to determine the best moves for a player based on a given board state. It returns a list of possible next board states after the player's move. The effectiveness of the code can be evaluated by testing it against various board configurations and evaluating the suggested moves.

9. Conclusion:
In conclusion, the code provides a framework for playing a two-player board game, and it incorporates advanced AI decision-making through the minimax algorithm with alpha-beta pruning. It is capable of evaluating game states and providing recommendations for the next best move. The code reflects effective problem formulation, strategic design decisions, and the handling of complex game rules and logic.

# Problem 2:

### Description:
The problem involves a text classification task, specifically a binary classification problem. The goal is to classify text data into two classes: "truthful" and "deceptive". This problem is commonly known as sentiment analysis, where the sentiment of a given text is classified as either positive or negative. In this case, "truthful" corresponds to a positive sentiment, and "deceptive" corresponds to a negative sentiment.

### Program Workflow:

- load_file(filename): This function loads the data from a given file, which is expected to be in a specific format. Each line in the file contains a label (either "truthful" or "deceptive") followed by the text data to be classified. The function reads the file, splits each line into a label and the corresponding text, and returns a dictionary containing the labels, text data, and a list of unique classes.

- preprocess_data(data): This function takes a text data as input and performs several preprocessing steps:
    1. Converts the text to lowercase.
    2. Splits the text into words.
    3. Removes common stop words from the text.
    4. Performs lemmatization to reduce words to their base forms.

- train_bayes_classifier(train_data): This function trains a naive Bayes classifier on the training data. It calculates word frequencies for both "truthful" and "deceptive" classes and builds a vocabulary of unique words. The classifier uses Laplace smoothing with a smoothing factor of 1.

- classifier(train_data, test_data): This function applies the trained naive Bayes classifier to the test data. For each text entry in the test data, it calculates the log probabilities of it belonging to the "truthful" and "deceptive" classes based on the word frequencies and prior probabilities. It then classifies the text based on the class with the higher log probability.

- The main part of the code reads command-line arguments to specify the training and test files. It checks that the number of classes in the training and test data is 2, and that the classes match. It then evaluates the classifier's performance on the test data by comparing its predictions to the ground truth labels.

### Problems, Assumptions, Simplifications, and Design Decisions:

1. Assumptions:
    - The data files are assumed to be in the specified format with labels and text on each line.
    - The binary classification problem is assumed to be sentiment analysis, with "truthful" corresponding to a positive sentiment and "deceptive" to a negative sentiment.

2. Simplifications:
    - The code uses simple preprocessing techniques like lowercase conversion, stop word removal, and lemmatization. More advanced natural language processing techniques could be employed for improved accuracy.
    - The code assumes binary classification and does not handle multi-class problems.

3. Design Decisions:
    - The code uses a naive Bayes classifier, a simple but effective probabilistic model for text classification.
    - Laplace smoothing with a smoothing factor of 1 is used to prevent zero probabilities for words not seen in the training data.
    - The code calculates and prints the classification accuracy as the evaluation metric.
