import csv
import random

def list_split(list_,n_parts):
    k, m = divmod(len(list_), n_parts)
    return [list_[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

def chunks_generation(label_path, truth_path, num_chunks = 10):
    #get all items
    e2wl, _, _ = gete2wlandw2el(label_path)
    items_without_ground_truths = list(e2wl.keys())
    items_count = len(items_without_ground_truths)
    #get items with ground truth
    items_with_ground_truths = []
    ground_truths = {}
    f = open(truth_path, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        item, truth = line
        ground_truths[item] = truth
        if int(truth)>=0:
            items_with_ground_truths.append(item)
            items_without_ground_truths.remove(item)

    random.shuffle(items_without_ground_truths)
    random.shuffle(items_with_ground_truths)

    items_without_ground_truths_splited = list_split(items_without_ground_truths, n_parts=num_chunks)
    items_with_ground_truths_splited = list_split(items_with_ground_truths, n_parts=num_chunks)
    chunks = []
    for i in range(num_chunks):
        chunk = items_with_ground_truths_splited[i] + items_without_ground_truths_splited[i]
        random.shuffle(chunk)
        chunks.append(chunk)
    return chunks




def gete2wlandw2el(datafile):
    e2wl = {}
    w2el = {}
    label_set = []

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, worker, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker, label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example, label])

        if label not in label_set:
            label_set.append(label)

    return e2wl, w2el, label_set




def getaccuracy(truthfile, truths, chunk = None):
    ground_truths = {}
    f = open(truthfile,'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        item, truth = line
        ground_truths[item] = truth

    ccount, tcount = 0,0
    for item, ground_truth in ground_truths.items():
        if chunk is not None:
            #for online evaluation, skip if the chunk does not contain the item
            if item not in chunk:
                continue
        if int(ground_truth) < 0:
            continue
        if truths.get(item) is None:
            continue
        aggregated_truth = truths[item]
        tcount += 1
        if aggregated_truth == ground_truth:
            ccount += 1
    return ccount * 1.0 / tcount

if __name__ == '__main__':
    #test
    e2wl, w2el, label_set = gete2wlandw2el("data/crowdscale2013/fact_eval/label.csv")
