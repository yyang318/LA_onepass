import math
import random
from collections import defaultdict
from data_pipeline import gete2wlandw2el,getaccuracy,chunks_generation
import time
from results_summary import OnlineResultsSummary
from LA_methods.baselines.sbic_online import sbic_online


if __name__ == '__main__':
    datasets = [

        ('active-crowd-toolkit/SP', 'SP'),
        ('active-crowd-toolkit/SP_amt', 'SP_amt'),
        ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
        ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
        ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
        ('crowd_truth_inference/d_jn-product', 'product'),
        ('crowd_truth_inference/d_sentiment', 'tweet'),
        ('SpectralMethodsMeetEM/bluebird', 'bird'),
        ('SpectralMethodsMeetEM/rte', 'rte'),
        ('SpectralMethodsMeetEM/trec', 'trec'),

    ]


    results = OnlineResultsSummary('sbic_online')
    num_chunks = 10
    num_rounds = 20
    for dataset, abbrev in datasets:
        random.seed(1)
        print(dataset, abbrev)
        one_pass_acc_list, two_pass_acc_list = [], []
        label_path = "data/" + dataset + "/label.csv"
        truth_path = "data/" + dataset + "/truth.csv"
        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for round in range(num_rounds):
            chunks = chunks_generation(label_path, truth_path, num_chunks)
            truths, progressive_accuracies, chunk_runtimes = sbic_online(e2wl, chunks=chunks, truthfile=truth_path)
            results.add(abbrev, progressive_accuracies, chunk_runtimes)

    print('SBIC accuracy')
    print(results.to_dataframe_accuracy_all_mean())

    print('SBIC runtime')
    print(results.to_dataframe_runtime_all_mean())
