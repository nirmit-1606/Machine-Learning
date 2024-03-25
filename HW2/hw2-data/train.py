#!/usr/bin/env python3

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
from svector import svector

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def filter_n_count_words(trainfile, n):
    word_counts = {}
    with open(trainfile, 'r') as file:
        for line in file:
            _, words = line.strip().split("\t")
            for word in words.split():
                word_counts[word] = word_counts.get(word, 0) + 1

    filtered_trainfile = "filtered_train.txt"  # Save the filtered data to a new file
    with open(trainfile, 'r') as infile, open(filtered_trainfile, 'w') as outfile:
        for line in infile:
            label, words = line.strip().split("\t")
            filtered_words = [word for word in words.split() if word_counts.get(word, 0) > n]
            if filtered_words:  # Only write lines with non-empty word lists
                outfile.write(f"{label}\t{' '.join(filtered_words)}\n")

    return filtered_trainfile

def make_vector(words):
    v = svector()
    v['<bias>'] = 1  # Add the bias term with a value of 1
    for word in words:
        v[word] += 1
    return v
    
def test(devfile, model):
    err = 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now
            
def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def avg_train(trainfile, devfile, testfile, filter_count = 1, epochs=5):
    t = time.time()
    if(filter_count > 0):
        trainfile = filter_n_count_words(trainfile, filter_count)
    best_err = 1.
    best_ep = 1
    model = svector()       # weight
    au_model = svector()    # auxiliary weight
    c = 0                   # counter
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent           # w <- w + yx
                au_model += c * label * sent    # wa <- wa + cyx
            c += 1
        dev_err = test(devfile, c * model - au_model)
        # best_err = min(best_err, dev_err)
        if dev_err < best_err:
            best_err = dev_err
            best_ep = it
        # print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    deploy(testfile, c * model - au_model)
    # Get top features
    # features_and_weights = best_model.items()
    # # For most positive features (descending order)
    # top_positive_features = sorted(features_and_weights, key=lambda x: x[1], reverse=True)[:20]
    # # For most negative features (ascending order)
    # top_negative_features = sorted(features_and_weights, key=lambda x: x[1])[:20]
    # print("\nTop 20 positive features")
    # print("Feature\t\tWeight")
    # for fe, we in top_positive_features:
    #     print(fe, "\t", we)
    # print("\nTop 20 negative features")
    # print("Feature\t\tWeight")
    # for fe, we in top_negative_features:
    #     print(fe, "\t", we)

    # predictions = []
    # examples = []

    # for label, words in read_from(devfile):
    #     prediction = best_model.dot(make_vector(words))
    #     predictions.append(prediction)
    #     examples.append((label, words))

    # # Create a list of tuples containing the example and its prediction score
    # examples_predictions = list(zip(examples, predictions))

    # # Sort the examples the prediction score in descending order
    # examples_predictions.sort(key=lambda x: x[1], reverse=True)

    # top_wrong_pos = []

    # for (label, words), prediction in examples_predictions:
    #     # Select examples where the model's prediction is strongly positive but the true label is negative
    #     if prediction > 0 and label == -1:
    #         top_wrong_pos.append(((label, words), prediction))

    # # Sort the examples the prediction score in ascending order
    # examples_predictions.sort(key=lambda x: x[1])

    # top_wrong_neg = []

    # for (label, words), prediction in examples_predictions:
    #     # Select examples where the model's prediction is strongly negative but the true label is positive
    #     if prediction < 0 and label == 1:
    #         top_wrong_neg.append(((label, words), prediction))

    # # Print or examine the selected examples
    # print("\n5 negative examples in dev where model most strongly believes to be positive")
    # for (label, words), prediction in top_wrong_pos[:5]:
    #     review = " ".join(words)
    #     print(f"True Label: {label}, Prediction: {prediction}, Example: {review}")
    # print("\n5 positive examples in dev where model most strongly believes to be negative")
    # for (label, words), prediction in top_wrong_neg[:5]:
    #     review = " ".join(words)
    #     print(f"True Label: {label}, Prediction: {prediction}, Example: {review}")

    print("best epoch %d best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_ep, best_err * 100, len(model), time.time() - t))
    # return best_model

def naive_avg_train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()       # weight
    s_model = svector()     # weight sum
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent   # w <- w + yx
            s_model += model            # ws <- ws + w
        dev_err = test(devfile, s_model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def deploy(testfile, model):
    deployment_file = "test.txt.predicted"
    pos_count = 0
    neg_count = 0
    with open(deployment_file, 'w') as outfile:
        for _, (_, words) in enumerate(read_from(testfile), 1): # note 1...|D|  
            p_label = '?' 
            if model.dot(make_vector(words)) > 0:
                p_label = '+'
                pos_count += 1
            else:
                p_label = '-'
                neg_count += 1
            outfile.write(f"{p_label}\t{' '.join(words)}\n")
    print(f"Positive count: {pos_count} Negative count: {neg_count}")

if __name__ == "__main__":
    # print("Vanilla Perceptron")
    # train(sys.argv[1], sys.argv[2], 10)
    # print("Average Perceptron")
    # for n in range(int(sys.argv[3])+1):
    #     print("Filtering %d count words" % (n))
    #     avg_train(sys.argv[1], sys.argv[2], n, 12)
    # print("Naive Average Perceptron")
    # naive_avg_train(sys.argv[1], sys.argv[2], 10)
    avg_train(sys.argv[1], sys.argv[2], sys.argv[3], filter_count=int(sys.argv[4]), epochs=int(sys.argv[5]))
    # deploy(sys.argv[3], best_model)
