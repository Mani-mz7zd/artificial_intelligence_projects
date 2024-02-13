###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids: Manikanta Kodandapani Naidu: k11, Pothapragada Venkata SG Krishna Srikar: vpothapr, G Vivek Reddy: gvi
#



import random
import math
from collections import defaultdict


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        log_prob = 0.0
        if model == "Simple":
            for i in range(len(sentence)):
                word = sentence[i]
                tag = label[i]
                log_prob += math.log(self.emission_probability(word, tag)) + math.log(self.prior_pos_prob[tag]/sum(self.prior_pos_prob.values()))
            
            return log_prob
        
        elif model == "HMM":
            prob_s1 = math.log(self.prior_pos_prob[label[0]] / sum(self.prior_pos_prob.values()), 10)
            z = 0
            t = 0
            for i in range(len(sentence)):
                z += math.log(self.emission_probability(sentence[i], label[i]), 10)
                if i != 0:
                    t += math.log(self.transition_probability(label[i - 1], label[i]), 10)
                    
            return prob_s1 + z + t
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):

        words_frequency_dict = defaultdict(lambda: defaultdict(float))
        prior_words_count = defaultdict(int)
        prior_pos_count = defaultdict(int)
        previous_pos_count = defaultdict(lambda: defaultdict(int))

        for sentence in data:
            for i in range(len(sentence[0])):
                words_frequency_dict[sentence[0][i]][sentence[1][i]] += 1

                prior_words_count[sentence[0][i]] += 1
                prior_pos_count[sentence[1][i]] += 1

                if i > 0:
                    previous_pos_count[sentence[1][i-1]][sentence[1][i]] += 1

        for key_word in words_frequency_dict.keys():
            words_frequency_dict[key_word] = dict(sorted(words_frequency_dict[key_word].items(), key=lambda x:x[1], reverse = True))

        self.words_data = words_frequency_dict
        self.prior_words_prob = prior_words_count
        self.prior_pos_prob = prior_pos_count
        self.previous_pos_prob = previous_pos_count
        
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    # Simple Bayes net where the parts of speech are independent of each other
    def simplified(self, sentence):
        parts_of_speech = ['']*len(sentence)

        for i in range(len(sentence)):
            p_val = 0
            for key in self.prior_pos_prob.keys():
                prob_value = self.get_bayes_probability(sentence[i], key)

                if prob_value>p_val:
                    parts_of_speech[i] = key
                    p_val = prob_value

        return parts_of_speech
    
    # HMM algorithm implementation
    def hmm_viterbi(self, sentence):
        viterbi_table = [{}]
        viterbi_track = {}
       
        # Probabilities for the first level because we do not have the transition probabilities at this level
        for pos in self.prior_pos_prob.keys():
            viterbi_table[0][pos] = self.get_bayes_probability(sentence[0], pos)
            viterbi_track[pos] = [pos]
     
        # Probabilities from the second level onwards
        for successor_level in range(1, len(sentence)):
            viterbi_table.append({})
            current_path = {}
            
            # Current pos for which we want to calculate the probability
            for current_pos in self.prior_pos_prob.keys():
                max_value = 0
                # Calculating for all the parts of speech because the previous part of speech can be anything.
                # Then we will take the maximum value form all the parts of speech and assign to the current cell
                for pre_pos in self.prior_pos_prob.keys():
                    value = viterbi_table[successor_level-1][pre_pos] * self.transition_probability(pre_pos,current_pos) * self.emission_probability(sentence[successor_level],current_pos)
                    if value > max_value:
                        max_value = value
                        state = pre_pos
                viterbi_table[successor_level][current_pos] = max_value
                current_path[current_pos] = viterbi_track[state] + [current_pos]

            viterbi_track = current_path

        max_value = -math.inf
        last_level = len(sentence) - 1
        # Taking the maximum probability and the corresponding part of speech
        # And determine the path that must be taking based on the best part of speech found at the last level
        for pos in self.prior_pos_prob.keys():
            if viterbi_table[last_level][pos] >= max_value:
                max_value  = viterbi_table[last_level][pos]
                best_state = pos
        state = best_state
        
        return viterbi_track[state]

    def transition_probability(self, pos1, pos2):
        if pos1 in self.previous_pos_prob and pos2 in self.previous_pos_prob[pos1]:
            return (self.previous_pos_prob[pos1][pos2]/sum(self.previous_pos_prob[pos1].values()))
        
        return 0.0000001
    
    def emission_probability(self, word, pos):
        exist = False
        if word in self.words_data.keys():
            if pos in self.words_data[word].keys():
                if pos in self.previous_pos_prob.keys():
                    exist = True
                    word_tag_prob = self.words_data[word][pos] / sum(self.previous_pos_prob[pos].values())
        if not exist:
            word_tag_prob =  self.grammar_rules(word, pos)

        return word_tag_prob

    
    def get_bayes_probability(self, word, pos):
        exist = False
        if pos in self.prior_pos_prob.keys():
            initial_tag_prob =  self.prior_pos_prob[pos]/sum(self.prior_pos_prob.values())
        else:
            initial_tag_prob = 0.00000001

        if word in self.words_data.keys():
            if pos in self.words_data[word].keys():
                if pos in self.previous_pos_prob.keys():
                    exist = True
                    word_tag_prob = self.words_data[word][pos] / sum(self.previous_pos_prob[pos].values())
        if not exist:
            word_tag_prob =  self.grammar_rules(word, pos)

        return word_tag_prob*initial_tag_prob
    
    def grammar_rules(self, word, tag):
        p = 0.9
        if word not in self.words_data.keys():
            if (list(word)[-3:] == list("ing") or list(word)[-2:] == list("ed") or list(word)[-3:] == list("ify")) and tag == 'verb':
                return p

            if (list(word)[-4:] == list("like") or list(word)[-4:] == list("less") or list(word)[-4:] == list("able") 
                or list(word)[-3:] == list("ful") or list(word)[-3:] == list("ous")or list(word)[-3:] == list("ish") 
                or list(word)[-2:] == list("ic") or list(word)[-3:] == list("ive")) and tag == 'adj':
                return p

            if (list(word)[-2:] == list("ly") ) and tag == 'adv':
                return p

            if (list(word)[-2:] == list("'s") or list(word)[-3:] == list("ist") or list(word)[-3:] == list("ion") 
                or list(word)[-4:] == list("ment"))and tag == 'noun':
                return p

            if tag == 'noun':
                return 0.4
            try:
                if int(word):
                    if tag == 'num':
                        return 1
            except ValueError:
                pass
        return 0.0000001


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

