import random
from data_pipeline import gete2wlandw2el,getaccuracy
import time
from results_summary import OfflineResultsSummary
from LA_methods.baselines.mv_offline import mv


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
        ('SpectralMethodsMeetEM/trec', 'trec')
    ]



    results = OfflineResultsSummary('mv')

    round = 20
    # random.seed(1)
    for dataset, abbrev in datasets:
        random.seed(1)
        print(dataset, abbrev)
        label_path = "data/"+dataset+"/label.csv"
        truth_path = "data/"+dataset+"/truth.csv"

        e2wl, w2el, label_set = gete2wlandw2el(label_path)
        for r in range(round):
            t1 = time.time()
            truths, a = mv(e2wl,label_set)
            t2 = time.time()

            mv_acc = getaccuracy(truth_path, truths)

            results.add(abbrev,mv_acc,t2-t1,1)

    print('MV results')
    print(results.to_dataframe_mean())