import math
import random
from collections import defaultdict
from data_pipeline import gete2wlandw2el,getaccuracy,chunks_generation
import time
from results_summary import OnlineResultsSummary




def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)

def sig(x):
    return 1/ (1+math.exp(-x))


def iCRH(e2wl, label_set, alpha=1, chunks = None,truthfile = None):
    progressive_accuracies = []
    chunk_runtimes = []

    t1 = time.time()
    start_processing = False

    source_accu_dists = defaultdict(lambda : 0.0)
    truths = {}

    for chunk in chunks:
        t2 = time.time()
        for item in chunk:
            source_weights = defaultdict(lambda: 1.0)
            weight_max = -1.0
            # weight_sum = 0.0
            # compute item truth
            item_votes = {}
            for class_ in label_set:
                item_votes[class_] = 0
            for worker, worker_label in e2wl[item]:
                item_votes[worker_label] += source_weights[worker]
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
            # update source accuracy distance after truths are estiamted
            for worker, label in e2wl[item]:
                if label == truths[item]:
                    source_accu_dists[worker] = source_accu_dists[worker] * alpha + 1
                if source_accu_dists[worker] > weight_max:
                    weight_max = source_accu_dists[worker]
            for worker, _ in e2wl[item]:
                source_weights[worker] = - math.log(source_accu_dists[worker]/weight_max + + 0.0000001) + + 0.0000001

        t3 = time.time()
        chunk_accuracy = getaccuracy(truthfile, truths)
        progressive_accuracies.append(chunk_accuracy)

        if start_processing:
            chunk_runtimes.append(t3 - t2)
        else:
            chunk_runtimes.append(t3 - t1)
        start_processing = True
    return truths, progressive_accuracies, chunk_runtimes, source_weights

