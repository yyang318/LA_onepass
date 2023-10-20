from LA_methods.baselines.tilcc import kmeans
from LA_methods.baselines.tilcc.CRH import run_CRH
from LA_methods.baselines.tilcc.generate_feature import generate_CRH_feature
from results_summary import OfflineResultsSummary
import random


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
        ('crowd_truth_inference/d_sentiment', 'senti_1k'),
        ('SpectralMethodsMeetEM/bluebird', 'bird'),
        ('SpectralMethodsMeetEM/rte', 'rte'),
        ('SpectralMethodsMeetEM/trec', 'trec'),


    ]


    results = OfflineResultsSummary('TILCC')

    num_rounds = 20
    for dataset, abbrev in datasets:
        random.seed(1)
        for round in range(num_rounds):
            print('dataset',abbrev,'round',round+1)
            t_CRH = run_CRH(dataset)
            t_feature = generate_CRH_feature(dataset)
            # acc = kmeans_pure.kmeans(dataset, "feature")
            acc, t_kmean, it = kmeans.kmeans(dataset, "feature")
            time_run = t_CRH + t_feature + t_kmean

            results.add(abbrev,acc,time_run,it+10) # +10 is the iterations of CRH for generating features

    print('TILCC results')
    print(results.to_dataframe_mean())
