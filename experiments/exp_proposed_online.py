import random
from data_pipeline import gete2wlandw2el,getaccuracy,chunks_generation
import time
from results_summary import OnlineResultsSummary
from LA_methods.proposed.proposed_method_online import one_and_two_pass
random.seed(1)

if __name__ == '__main__':

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
        ('mill','mill'),

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

    results_onepass = OnlineResultsSummary('onepass')
    results_twopass = OnlineResultsSummary('twopass')
    random.seed(1)
    round = 20
    num_chunks = 10
    for dataset, abbrev in datasets:
        print(dataset, abbrev)
        label_path = 'data/' + dataset + "/label.csv"
        truth_path = 'data/' + dataset + "/truth.csv"

        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for r in range(round):
            chunks = chunks_generation(label_path, truth_path, num_chunks)
            onepass_truths, twopass_truths, \
            progressive_onepass_accuracies, progressive_twopass_accuracies, \
            onepass_chunk_runtimes, twopass_chunk_runtimes = one_and_two_pass(e2wl, w2el, label_set, chunks, truth_path)
            results_onepass.add(abbrev, progressive_onepass_accuracies, onepass_chunk_runtimes)
            results_twopass.add(abbrev, progressive_twopass_accuracies, twopass_chunk_runtimes)

    print('Onepass accuracy')
    print(results_onepass.to_dataframe_accuracy_all_mean())
    print('Twopass accuracy')
    print(results_twopass.to_dataframe_accuracy_all_mean())


    print('Onepass runtime')
    print(results_onepass.to_dataframe_runtime_all_mean())
    print('Twopass runtime')
    print(results_twopass.to_dataframe_runtime_all_mean())
