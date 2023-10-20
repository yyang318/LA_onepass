import random
from data_pipeline import gete2wlandw2el,getaccuracy,chunks_generation
import time
from results_summary import OnlineResultsSummary
random.seed(1)

def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)

def mv(e2wl, label_set):
    items = list(e2wl.keys())
    votes = {}
    truths={}


    for item in items:
        #aggregate truth
        item_votes = {}
        for class_ in label_set:
            item_votes[class_] = 0
        for worker, worker_label in e2wl[item]:
            item_votes[worker_label]  = item_votes[worker_label] + 1
        candidate = []
        max_ = -1
        for class_ in label_set:
            if item_votes.get(class_) is None:
                continue
            if item_votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = item_votes.get(class_)
            elif item_votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)
        for class_ in label_set:
            item_votes[class_] = item_votes[class_] / len(e2wl[item])
        votes[item] = item_votes

    return truths, votes

def one_and_two_pass(e2wl, w2el, label_set,chunks,truthfile):
    progressive_twopass_accuracies = []
    twopass_chunk_runtimes = []

    progressive_onepass_accuracies = []
    onepass_chunk_runtimes = []
    t1 = time.time()
    start_processing = False
    alpha, beta = 2, 2
    m = len(w2el)
    n = len(e2wl)
    K = len(label_set)
    c={}
    t={}
    a={}
    onepass_truths={}
    twopass_truths = {}
    for worker in w2el.keys():
        c[worker] = alpha - 1
        t[worker] = alpha + beta - 2
        a[worker] = c[worker] / t[worker]


    for chunk in chunks:
        t2 = time.time()
        for item in chunk:
            #aggregate truth
            votes = {}
            for worker, worker_label in e2wl[item]:
                if votes.get(worker_label) is None:
                    votes[worker_label] = 0
                votes[worker_label]  = votes[worker_label] + a[worker]
            candidate = []
            max_ = -1
            for class_ in label_set:
                if votes.get(class_) is None:
                    continue
                if votes.get(class_) > max_:
                    candidate.clear()
                    candidate.append(class_)
                    max_ = votes.get(class_)
                elif votes.get(class_) == max_:
                    candidate.append(class_)
            onepass_truths[item] = random.choice(candidate)

            #update ability
            for worker, worker_label in e2wl[item]:
                t[worker] = t[worker] + 1
                if worker_label == onepass_truths[item]:
                    c[worker] = c[worker] + 1
                a[worker] = c[worker] / t[worker]
        t3 = time.time()
        #onepass finished here
        #twopass starts here
        for item in chunk:
            votes = {}
            for worker, worker_label in e2wl[item]:
                if votes.get(worker_label) is None:
                    votes[worker_label] = 0
                votes[worker_label] = votes[worker_label] + (a[worker] * K - 1)
            candidate = []
            max_ = -999
            for class_ in label_set:
                if votes.get(class_) is None:
                    continue
                if votes.get(class_) > max_:
                    candidate.clear()
                    candidate.append(class_)
                    max_ = votes.get(class_)
                elif votes.get(class_) == max_:
                    candidate.append(class_)
            twopass_truths[item] = random.choice(candidate)
        t4 = time.time()


        onepass_chunk_accuracy = getaccuracy(truthfile, onepass_truths)
        twopass_chunk_accuracy = getaccuracy(truthfile, twopass_truths)

        progressive_onepass_accuracies.append(onepass_chunk_accuracy)
        progressive_twopass_accuracies.append(twopass_chunk_accuracy)


        if start_processing:
            onepass_chunk_runtimes.append(t3 - t2)
            twopass_chunk_runtimes.append(t4 - t2)
        else:
            onepass_chunk_runtimes.append(t3 - t1)
            twopass_chunk_runtimes.append(t4 - t1)
        start_processing = True
    return onepass_truths, twopass_truths, \
           progressive_onepass_accuracies, progressive_twopass_accuracies, \
           onepass_chunk_runtimes, twopass_chunk_runtimes



