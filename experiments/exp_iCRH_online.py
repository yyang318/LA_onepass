import random
from data_pipeline import gete2wlandw2el,getaccuracy,chunks_generation
import time
from results_summary import OnlineResultsSummary
from LA_methods.baselines.iCRH import iCRH
random.seed(1)

if __name__ == '__main__':

    data_path = '../data/'
    datasets = [
        ('crowdscale2013/sentiment', 'senti'),
        ('crowdscale2013/fact_eval', 'fact'),
        ('active-crowd-toolkit/CF', 'CF'),
        ('active-crowd-toolkit/CF_amt', 'CF_amt'),
        ('active-crowd-toolkit/MS', 'MS'),
        ('crowd_truth_inference/s4_Face Sentiment Identification', 'face'),
        ('crowd_truth_inference/s5_AdultContent', 'adult'),
        ('SpectralMethodsMeetEM/dog', 'dog'),
        ('SpectralMethodsMeetEM/web', 'web'),
        ('mill', 'mill'),

        ('active-crowd-toolkit/SP', 'SP'),
        ('active-crowd-toolkit/SP_amt', 'SP_amt'),
        ('active-crowd-toolkit/ZenCrowd_all', 'ZC_all'),
        ('active-crowd-toolkit/ZenCrowd_in', 'ZC_in'),
        ('active-crowd-toolkit/ZenCrowd_us', 'ZC_us'),
        ('crowd_truth_inference/d_jn-product', 'product'),
        ('crowd_truth_inference/d_sentiment', 'senti_1k'),
        ('SpectralMethodsMeetEM/bluebird', 'bird'),
        ('SpectralMethodsMeetEM/rte', 'rte'),
        ('SpectralMethodsMeetEM/trec', 'trec'),

    ]

    # random.seed(1)
    results = OnlineResultsSummary('iCRH')
    num_chunks = 10
    num_rounds = 20
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        one_pass_acc_list, two_pass_acc_list = [], []
        label_path = "data/" + dataset + "/label.csv"
        truth_path = "data/" + dataset + "/truth.csv"
        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for round in range(num_rounds):
            chunks = chunks_generation(label_path, truth_path, num_chunks)
            truths, progressive_accuracies, chunk_runtimes, source_weights = iCRH(e2wl, label_set, chunks=chunks,
                                                                                  truthfile=truth_path)
            results.add(abbrev, progressive_accuracies, chunk_runtimes)

    print('iCRH accuracy')
    print(results.to_dataframe_accuracy_all_mean())

    print('iCRH runtime')
    print(results.to_dataframe_runtime_all_mean())