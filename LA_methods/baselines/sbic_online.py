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



def sbic_online(e2wl, alpha=2, beta = 1, q = 0.5, chunks = None, truthfile = None):
    progressive_accuracies = []
    chunk_runtimes = []

    t1 = time.time()
    start_processing = False

    z = dict()
    z = defaultdict(lambda: math.log(q/(1 - q)), z)
    p = dict()
    p = defaultdict(lambda: alpha / (alpha + beta), p)
    truths = {}

    M_numerator_accumulated = dict()
    M_numerator_accumulated = defaultdict(lambda: alpha, M_numerator_accumulated)

    M_denominator_accumulated = dict()
    M_denominator_accumulated = defaultdict(lambda: alpha + beta, M_denominator_accumulated)



    for chunk in chunks:
        t2 = time.time()
        for item in chunk:
            item_labels = e2wl[item]
            random.shuffle(item_labels)
            for worker, label in item_labels:


                p_worker = (M_numerator_accumulated[worker]) / (M_denominator_accumulated[worker])

                z[item] += math.log(p_worker / (1 - p_worker)) if label == '1' \
                    else -math.log(p_worker / (1 - p_worker))
            # update worker quality after z[item] estimation is finished
            for worker, label in e2wl[item]:
                sig_param = z[item] if label == '1' else -z[item]
                sig_term = sig(sig_param)
                M_numerator_accumulated[worker] += sig_term
                M_denominator_accumulated[worker] += 1
            truths[item] = '1' if z[item] >= 0 else '0'
        t3 = time.time()
        chunk_accuracy = getaccuracy(truthfile, truths)
        progressive_accuracies.append(chunk_accuracy)

        if start_processing:
            chunk_runtimes.append(t3 - t2)
        else:
            chunk_runtimes.append(t3 - t1)
        start_processing = True
    return truths, progressive_accuracies, chunk_runtimes

