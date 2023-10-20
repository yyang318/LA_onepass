import random
import math
from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary

random.seed(1)

def mv_one_versus_all(e2wl, label_set, target):
    items = list(e2wl.keys())
    votes = {}
    truths={}


    for item in items:
        #aggregate truth
        item_positive_votes = 0
        for worker, worker_label in e2wl[item]:
            if worker_label == target:
                item_positive_votes  = item_positive_votes + 1
        votes[item] = item_positive_votes * 1.0 / len(e2wl[item])
        if votes[item] > 0.5:
            truths[item] = '1'
        elif votes[item] < 0.5:
            truths[item] = '0'
        else:
            truths[item] = random.choice(['1','0'])

    return truths, votes

def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)

def dict_values_allclose(dict1, dict2, rtol = 1e-3, atol=1e-08):
    for key in dict1.keys():
        if math.fabs(dict1[key] - dict2[key]) > atol + rtol * math.fabs(dict2[key]):
            return False
    return True

def from_z_to_truths_binary(z_i):
    truths = {}
    for item in z_i.keys():
        if z_i[item] > 0.5:
            truths[item] = '1'
        elif z_i[item] < 0.5:
            truths[item] = '0'
        else:
            truths[item] = random.choice(['1', '0'])
    return truths

def from_z_to_truths_multi(z_i, label_set):
    truths = {}
    for item in z_i.keys():
        votes = z_i.get(item)
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
        truths[item] = random.choice(candidate)
    return truths


def bwa(e2wl, w2el, label_set, lambda_=1, a_v=15):
    adj_coef = 4 * (1 - 1 / len(label_set))
    iterations = 0
    if len(label_set) == 2:
        #binary
        z_i, it_ = bwa_binary(e2wl,w2el,label_set,target='1', lambda_ = lambda_, a_v = a_v, adj_coef = adj_coef)
        truths = from_z_to_truths_binary(z_i)
        iterations += it_
    else:
        z_i = {}
        for target_label in label_set:
            z_ik, it_ = bwa_binary(e2wl,w2el,label_set,target=target_label,lambda_ = lambda_, a_v = a_v, adj_coef = adj_coef)
            for item in e2wl.keys():
                dict_ = z_i.get(item)
                if dict_ is None:
                    dict_ = {}
                    z_i[item] = dict_
                dict_[target_label] = z_ik[item]
            truths = from_z_to_truths_multi(z_i, label_set)
            iterations += it_
    return truths,z_i, iterations


def bwa_binary(e2wl,w2el,label_set,target='1', lambda_ = 1, a_v = 15, adj_coef = 1):
    mv_truths, z_i = mv_one_versus_all(e2wl,label_set,target=target)

    items = list(e2wl.keys())
    workers = list(w2el.keys())
    total_label_count = 0
    b_v = 0

    v_j = {}

    target_label = target
    for item in items:
        item_worker_count = len(e2wl[item])
        total_label_count += item_worker_count

        b_v = b_v + item_worker_count * z_i[item] * (1-z_i[item])
    b_v = a_v * b_v / total_label_count * adj_coef

    for it_ in range(500):
        last_z_i = z_i.copy()
        mu = list_mean(list(z_i.values()))

        #update worker quality
        for worker in workers:
            worker_error = 0
            for item, worker_label in w2el[worker]:
                worker_error += (1-z_i[item])**2 if worker_label == target_label else z_i[item]**2
            v_j[worker] = (a_v + len(w2el[worker])) / (b_v + worker_error)

        for item in items:
            weithed_labels, weights = 0, 0
            for worker, worker_label in e2wl[item]:
                weithed_labels += v_j[worker] if worker_label == target_label else 0
                weights += v_j[worker]
            z_i[item] = (lambda_ * mu + weithed_labels) / (lambda_ + weights)
        if dict_values_allclose(last_z_i,z_i):
            break
    return z_i, it_+1


