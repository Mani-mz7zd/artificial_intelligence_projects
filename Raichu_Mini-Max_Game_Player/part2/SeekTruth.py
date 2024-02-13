# SeekTruth.py : Classify text objects into two categories
#
# Manikanta Kodandapani Naidu: k11, G Vivek Reddy: gvi, Pothapragada Venkata SG Krishna Srikar: vpothapr
#

import sys
import math
from collections import defaultdict


def lemmatize(word):
    suffixes = {
        'ies$':'y',
        'sses$': 'ss',
        'ss$': 'ss',
        's$':'',
        'ing$': '',
        'ed$': '',
        'er$': '',
        'ly$': '',
        'ation$': 'ate',
        'ment$': '',
        'ness$': '',
        'ize$': ''  
    }

    for pattern, replacement in suffixes.items():
        if word.endswith(pattern[:-1]):  
            return word[:-len(pattern) + 1] + replacement
    return word


def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}


def preprocess_data(data):
    # Convert text to lowercase
    data = data.lower()
    # splitting the string into words
    words = data.split()
    
    # Tokenization: Split text into words
    # words = re.findall(r"(?u)\b\w\w+\b", data)
    
    # Removing stop words    
    stop_words = set(['a', 'are', 'won', 'out', 'about', 'down', 'her', 'too', 'now', 'there', 
               'having', 'at', 'what', 'as', 'shouldn', 'doing', "wouldn't", 'above', 'most', 'under', "mightn't", 
               'were', "shouldn't", 'if', 'been', 'off', 'how', 'hadn', 'isn', 'their', 'these', "wasn't", 'you', 
               'with', 't', 'aren', 'ma', 'to', 'your', 'this', 'such', 'our', 'why', 'herself',
                'o', 'himself', 'where', 'i', 'weren', 'for', 'more', 'further', 'again', 'couldn', "won't", 
                'ours', 'over', 'nor', 'only', "aren't", 'did', "doesn't", 'he', 'because', 'just', 'was', 'or', 
                'all', 'me', 'them', 'being', 'yourselves', "that'll", "needn't", 'can', 'the', 'have', 'but', 
                'his', 'some', 'its', 'both', 'any', 'until', "you're", "weren't", 'wouldn', "you'll", 'not',
                "hasn't", 'don', 'yourself', 'and', "she's", 'myself', 'few', 'had', 'y', 'haven', 'very', 'before',
                'is', 'than', 'be', 'from', 'into', 'do', 'theirs', 'hers', "couldn't", 've', 'between', 'shan', 'no', 
                'same', 'we', "don't", 'once', "isn't", 'my', 'on', 'ain', 'whom', 'wasn', 'needn', "should've", "it's", 
                'mustn', 'didn', 'doesn', 'an', 'has', 'those', 'so', 'after', "hadn't", 'when', 'itself', 
                'while', 'that', 'him', 'of', 'here', 'm', 'during', 'own', 'who', 'does', 'd', "didn't", 'mightn',
                'each', "you've", 'by', 're', 'through', 'below', 'ourselves', 'in', 'themselves', 'then', 's', 'will', 
                'up', 'against', 'other', "mustn't", 'it', "you'd", 'hasn', 'should', 'which', 'am', "shan't", "haven't", 'they', 'she', 'yours', 'll','I','my'])
    

    words = [word for word in words if word not in stop_words]
    
    # Remove punctuation and special characters
    # words = [word for word in words if word.isalpha()]

    # lemmatization
    lemmatized_words = [lemmatize(word) for word in words]
    
    return lemmatized_words


def train_bayes_classifier(train_data):
    vocab = set()
    word_frequencies_truthful = defaultdict(int)
    word_frequencies_deceptive = defaultdict(int)

    for i in range(len(train_data["objects"])):
        words_list = preprocess_data(train_data["objects"][i])
        vocab.update(words_list)
        label = train_data["labels"][i]

        word_frequencies = word_frequencies_truthful if label == "truthful" else word_frequencies_deceptive

        for word in words_list:
            word_frequencies[word] += 1

    return word_frequencies_truthful, word_frequencies_deceptive, vocab


def classifier(train_data, test_data):
    word_frequencies_truthful, word_frequencies_deceptive, vocab = train_bayes_classifier(train_data)

    prior_probability_truthful = len([i for i in train_data["labels"] if i == 'truthful']) / len(train_data["labels"])
    prior_probability_deceptive = 1 - prior_probability_truthful

    results = []

    for entry in test_data["objects"]:
        words = preprocess_data(entry)
        log_prob_truth = math.log(prior_probability_truthful)
        log_prob_decept = math.log(prior_probability_deceptive)

        for word in words:
            if word in vocab:
                # Laplace smoothing with a smoothing factor of 1
                log_prob_truth += math.log((word_frequencies_truthful[word] + 1) / (sum(word_frequencies_truthful.values()) + len(vocab)))
                log_prob_decept += math.log((word_frequencies_deceptive[word] + 1) / (sum(word_frequencies_deceptive.values()) + len(vocab)))

        if log_prob_truth > log_prob_decept:
            results.append("truthful")
        else:
            results.append("deceptive")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    train_data = load_file(train_file)
    test_data = load_file(test_file)

    if sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2:
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}
    results = classifier(train_data, test_data_sanitized)

    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    accuracy = 100.0 * correct_ct / len(test_data["labels"])
    print("Classification accuracy = %5.2f%%" % accuracy)
