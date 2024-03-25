#!/usr/bin/env python
import sys
import time
from svector import svector
import heapq
import pickle

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    if isinstance(words, svector): # yilin: for caching dataset
        return words

    v = svector()
    v['BIAS'] = 1.
    for word in words:
        v[word] += 1
    return v


def test(devfile, model, cache = False):
    tot, err = 0, 0
    if not cache:           # kaibo: if cache, devfile is cached data instead of file name, no need to read again
        devfile = read_from(devfile)
    for i, (label, words) in enumerate(devfile, 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def show_feature_example(devfile, model):
    # kaibo: show most positive/negative features
    a = sorted([(v,k) for k,v in model.items()])
    for i,(x,y) in enumerate(a[:-21:-1],1): print('%d. %s, %s' % (i,y,x))
    for i,(x,y) in enumerate(a[:20],1): print('%d. %s, %s' % (i,y,x))

    # kaibo: show most wrongly predicted positive/negative examples
    dev_predicts = []
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        predict = model.dot(make_vector(words))
        if label * predict <= 0:
            dev_predicts.append((predict, ' '.join(words)))
    dev_predicts.sort()
    for i,(_, line) in enumerate(dev_predicts[:-6:-1],1):
        print('Top %d wrongly predict +: %s' % (i, line))
    for i,(_, line) in enumerate(dev_predicts[:5], 1):
        print('Top %d wrongly predict -: %s' % (i, line))

def neglect(data, num_p):   # kaibo: data and new_data has the same format as cache_vector()'s output
    if num_p == 0: return data
    new_data = []
    vocab = svector()       # kaibo: dict of the whole training set, not a single line
    for _, vec in data:     # kaibo: merge all lines of dicts into the whole vocabulary
        vocab += vec
    for label, vec in data:
        v = svector()
        for word, n in vec.items():
            if vocab[word] > num_p:
                v[word] = n
        if v:
            new_data.append((label, v))
    return new_data


def cache_vector(textfile): #kaibo: return a list of [(label, sentence_dict)] * lines
    return [(label, make_vector(words)) for label,words in read_from(textfile)]

def train_perc(trainfile, devfile, epochs=5, cache = False, pruning = 0):
    if cache:                                     # kaibo: if cache, update devfile as cached data instead of file name
        #traindata = list(read_from(trainfile))                 # 1.naive caching with word list by line, 2.6s -> 2.1s
        #devfile = list(read_from(devfile))
        traindata = neglect(cache_vector(trainfile),pruning)    # 2.fast caching with word vector/dict by line, 2.6s -> 1.2s
        devfile = cache_vector(devfile)

    t = time.time()
    best_err = best_err_ave = model_best = 1.
    model = svector()
    model_adjust = svector()
    c = 1
    dev_err = 1

    for it in range(1, epochs+1):
        updates = 0
        if not cache:
            traindata = read_from(trainfile)
        for i, (label, words) in enumerate(traindata, 1): # label is +1 or -1
            sent = words if cache else make_vector(words)   # if cache, words is already vector/dict
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_adjust +=  c * label * sent
            c += 1
        dev_err = test(devfile, model, cache = cache)
        dev_err_ave = test(devfile, model *c - model_adjust, cache = cache)
        best_err = min(best_err, dev_err)

        if dev_err_ave < best_err_ave:
            best_err_ave = dev_err_ave
            model_best = model * c - model_adjust
        print("epoch %d, update %.1f, dev %.1f, dev_ave %.1f" % (it, updates / i * 100, dev_err * 100, dev_err_ave * 100))

    print("best dev err %.1f, best dev_ave err %.1f, |w|=%d, time: %.1f secs" % (best_err * 100, best_err_ave * 100, len(model), time.time() - t))

    dev_predictions(devfile, model_best)

    return model_best

def dev_predictions(filename, model):
    dev_file = filename
    dev_wd_pred = 'oneHot_dev_pred.txt'
    with open(dev_wd_pred, 'w') as file:
        for i, (_, words) in enumerate(dev_file, 1):
            pred = '+' if model.dot(make_vector(words)) > 0 else '-'
            # label = '+' if label > 0 else '-'
            new_words = [word for word in words if word != 'BIAS']
            file.write(f"{pred}\t{' '.join(new_words)}\n")

def run(file_train, file_dev, epochs = 5):

    ############### Q1.3+Q2 ################
    # print('%s' % ('-'*10 +'Q1.3+Q2' + '-'*10))
    # model = train_perc(file_train, file_dev, epochs)
    # show_feature_example(file_dev, model)

    # ############### Q2.5 ################
    # print('%s' % ('-'*10 + 'Q2.5' + '-'*10))
    # model = train_perc(file_train, file_dev, epochs, cache = True)

    ############### Q3.1 ################
    print('%s' % ('-'*10 + 'Q3.1 pruning 1-count words' + '-'*10))
    model = train_perc(file_train, file_dev, epochs, cache = True, pruning = 1)

    # ############### Q3.5 ################
    # print('%s' % ('-'*10 + 'Q3.5 pruning 2-count words' + '-'*10))
    # model = train_perc(file_train, file_dev, epochs, cache = True, pruning = 2)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], 10)
