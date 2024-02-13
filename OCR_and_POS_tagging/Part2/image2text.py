#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Manikanta Kodandapani Naidu: k11, Pothapragada Venkata SG Krishna Srikar: vpothapr, G Vivek Reddy: gvi


from PIL import Image, ImageDraw, ImageFont
import sys
from collections import defaultdict
import math
import operator

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

# Training text data parsing
def read_text(train_txt_fname):
    parsed_data = []
    with open(train_txt_fname, 'r', encoding="utf8") as file:
        for line in file:
            words = line.split()[::]
            formatted_line = [word + ' ' for word in words]
            parsed_data.append(formatted_line)

    return parsed_data

# Calculating emission probabilities
def get_emission_prob(test_letters, train_letters):
    emission_probs = defaultdict(lambda: defaultdict(float))
    test_points = get_black_points(test_letters)
    train_points = get_black_points(train_letters)

    for i in range(len(test_letters)):
        test_letter = test_letters[i]
        for letter, val in train_letters.items():
            black_score = 0
            white_score = 0
            total = CHARACTER_WIDTH * CHARACTER_HEIGHT
            for x in range(CHARACTER_HEIGHT):
                for y in range(CHARACTER_WIDTH):
                    if test_letter[x][y] == val[x][y] == "*":
                        black_score += 1

                    if test_letter[x][y] == val[x][y] == ' ':
                        white_score += 1

            black_ratio_test = test_points[0] / test_points[1]
            black_ratio_train = train_points[0] / train_points[1]

            if black_ratio_test > black_ratio_train:
                emission_probs[i][letter] = 0.6 * (black_score / total) + 0.4 * (white_score / total)
            else:
                emission_probs[i][letter] = 0.9 * (black_score / total) + 0.1 * (white_score / total)

    return emission_probs

# Calculating black - "*" appearances count
def get_black_points(letter_val):
    black_points = 0
    total_points = 0

    for row in letter_val:
        for char in row:
            total_points += 1
            if char == '*':
                black_points += 1

    return [black_points, total_points]

# Calculating initial letter probabilities
def get_initial_prob(train_txt_fname):
    data = read_text(train_txt_fname)
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    initial_letter_counts = defaultdict(int)

    for line in data:
        for word in line:
            first_letter = word[0]
            if first_letter in TRAIN_LETTERS:
                initial_letter_counts[first_letter] += 1

    # Initial letter probabilities
    initial_state_probs = {letter: count / sum(initial_letter_counts.values()) for letter, count in initial_letter_counts.items()}

    return initial_state_probs

# Calculating transition probabilities
def get_transition_prob(train_txt_fname):
    data = read_text(train_txt_fname)
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    transition_count = defaultdict(lambda: defaultdict(int))

    # Count transitions and total appearances for each letter
    for line in data:
        for word in line:
            for i in range(len(word) - 1):
                if word[i] in TRAIN_LETTERS and word[i + 1] in TRAIN_LETTERS:
                    transition_count[word[i]][word[i+1]] += 1

    transition_probabilities = defaultdict(lambda: defaultdict(float))
    for current_letter in transition_count:
        total_transitions = sum(transition_count[current_letter].values())
        transition_probabilities[current_letter] = {
            next_letter: (count) / total_transitions
            for next_letter, count in transition_count[current_letter].items()
        }

    # Calculate total transition probabilities - Normalization
    transitions_probabilities_total = defaultdict(lambda: defaultdict(int))
    trans_total = 0

    for letter in transition_probabilities:
        trans_total += sum(transition_probabilities[letter].values())

    for first_letter in transition_probabilities:
        for second_letter, prob in transition_probabilities[first_letter].items():
            transitions_probabilities_total[first_letter][second_letter] = prob / float(trans_total)

    return transitions_probabilities_total


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def simplified(test_letters, train_letters, emission_probs):
    recognized_letters = ['']*len(test_letters)

    # Selecting the letter with maximum emission probability
    for i in range(len(test_letters)):
        recognized_letters[i] = max(emission_probs[i], key=emission_probs[i].get)

    return "".join(recognized_letters)


def hmm_viterbi(test_letters, train_letters, emission_probs, transition_probs, initial_probs):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    recognized_letters = [''] * len(test_letters)
    viterbi_table = [[{'probability': 0, 'letter': ''} for i in range(len(test_letters))] for j in range(len(TRAIN_LETTERS))]

    # Initial state probability for the first letter
    for train_index in range(len(TRAIN_LETTERS)):
        if TRAIN_LETTERS[train_index] in initial_probs:
            viterbi_table[train_index][0] = {'probability': -math.log10(emission_probs[0][TRAIN_LETTERS[train_index]]), 'letter': TRAIN_LETTERS[train_index]}

    # HMM Probabilities for the rest of the letters 
    for test_index in range(1, len(test_letters)):
        for val in emission_probs[test_index].keys():    
            sub_viterbi = {}
            for train_index in range(len(TRAIN_LETTERS)):
                if TRAIN_LETTERS[train_index] in transition_probs: 
                    if val in transition_probs[TRAIN_LETTERS[train_index]] and viterbi_table[train_index][test_index - 1]['probability'] != 0:                
                        sub_viterbi[val] = -30 * math.log10(emission_probs[test_index][val]) - 0.01 * math.log10(
                            transition_probs[TRAIN_LETTERS[train_index]][val]) - 0.01 * math.log10(
                            viterbi_table[train_index][test_index - 1]['probability'])

            if sub_viterbi:
                final_letter = max(sub_viterbi.items(), key=operator.itemgetter(1))[0]
                viterbi_table[TRAIN_LETTERS.index(val)][test_index] = {'probability': sub_viterbi[final_letter], 'letter': final_letter}
            else:
                viterbi_table[TRAIN_LETTERS.index(val)][test_index] = {'probability': 0, 'letter': ''}

    # Traceback to find the best sequence of letters
    for test_index in range(len(test_letters)):
        min_cost = float('inf')
        for train_index in range(len(TRAIN_LETTERS)):
            if train_index < len(TRAIN_LETTERS) and viterbi_table[train_index][test_index]['probability'] < min_cost and viterbi_table[train_index][test_index]['probability'] != 0:
                min_cost = viterbi_table[train_index][test_index]['probability']
                recognized_letters[test_index] = TRAIN_LETTERS[train_index]

    return ''.join(recognized_letters)

#####
# main program
def main():
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

    (train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)

    ## Below is just some sample code to show you how the functions above work. 
    # You can delete this and put your own code here!
    emission_probabilities = get_emission_prob(test_letters, train_letters)
    initial_probabilities = get_initial_prob(train_txt_fname)
    transition_probabilities = get_transition_prob(train_txt_fname)

    # Each training letter is now stored as a list of characters, where black
    #  dots are represented by *'s and white dots are spaces. For example,
    #  here's what "a" looks like:
    # print("\n".join([ r for r in train_letters['a'] ]))

    # # Same with test letters. Here's what the third letter of the test data
    # #  looks like:
    # print("\n".join([ r for r in test_letters[2] ]))


    simplified_res = simplified(test_letters, train_letters, emission_probabilities)
    hmm_viterbi_res = hmm_viterbi(test_letters, train_letters, emission_probabilities, transition_probabilities, initial_probabilities)
    # The final two lines of your output should look something like this:
    print("Simple: " + simplified_res)
    print("   HMM: " + hmm_viterbi_res)

if __name__ == "__main__":
    main()