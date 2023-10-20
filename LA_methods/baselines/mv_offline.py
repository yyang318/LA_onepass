import random
from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary

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






