import csv
import math
import random
import time


class CRH:
    def __init__(self, e2wl, w2el, label_set, datatype, distancetype):
        self.e2wl = e2wl
        self.w2el = w2el
        self.weight = dict()
        self.label_set = label_set
        self.datatype = datatype
        self.distype = distancetype

    def distance_calculation(self, example, label):
        if self.datatype == 'continuous' and self.distype == 'normalized absolute loss':
            raise NotImplementedError('Continuous dataype not supported')

        elif self.datatype == 'continuous' and self.distype == 'normalized square loss':
            raise NotImplementedError('Continuous dataype not supported')

        elif self.datatype == 'categorical' and self.distype == '0/1 loss':
            if self.truth[example] != label:
                return 1.0
            else:
                return 0.0
        else:
            print('datatype or distancetype error!')

    def examples_truth_calculation(self):
        self.truth = dict()

        if self.datatype == 'continuous' and self.distype == 'normalized absolute loss':

            raise NotImplementedError('Continuous dataype not supported')

        elif self.datatype == 'continuous' and self.distype == 'normalized square loss':

            raise NotImplementedError('Continuous dataype not supported')


        elif self.datatype == 'categorical' and self.distype == '0/1 loss':

            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for worker, label in worker_label_set:
                    if (temp.__contains__(label)):
                        temp[label] = temp[label] + self.weight[worker]
                    else:
                        temp[label] = self.weight[worker]

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                if len(candidate) > 0:
                    self.truth[example] = random.choice(candidate)
                else:
                    self.truth[example] = random.choice(self.label_set)

        else:
            print('datatype or distancetype error!')

    def workers_weight_calculation(self):
        weight_max = 0.0

        self.weight = dict()

        for worker, example_label_set in self.w2el.items():
            dif = 0.0
            for example, label in example_label_set:
                dif = dif + self.distance_calculation(example, label)

            if dif == 0.0:
                dif = 0.00000001

            self.weight[worker] = dif
            if self.weight[worker] > weight_max:
                weight_max = self.weight[worker]

        for worker in self.w2el.keys():
            self.weight[worker] = self.weight[worker] / weight_max

        for worker in self.w2el.keys():
            self.weight[worker] = - math.log(self.weight[worker] + 0.0000001) + 0.0000001

    def Init_truth(self):
        self.truth = dict()
        self.std = dict()

        if self.datatype == 'continuous':
            raise NotImplementedError('Continuous dataype not supported')


        else:
            # using majority voting to obtain initial value
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                for _, label in worker_label_set:
                    if (temp.__contains__(label)):
                        temp[label] = temp[label] + 1
                    else:
                        temp[label] = 1

                max = 0
                for label, num in temp.items():
                    if num > max:
                        max = num

                candidate = []
                for label, num in temp.items():
                    if max == num:
                        candidate.append(label)

                self.truth[example] = random.choice(candidate)

    def get_e2lpd(self):

        if self.datatype == 'continuous':
            raise NotImplementedError('Continuous dataype not supported')
        else:
            e2lpd = dict()
            for example, worker_label_set in self.e2wl.items():
                temp = dict()
                sum = 0.0
                for worker, label in worker_label_set:
                    if (temp.__contains__(label)):
                        temp[label] = temp[label] + self.weight[worker]
                    else:
                        temp[label] = self.weight[worker]
                    sum = sum + self.weight[worker]

                for label in temp.keys():
                    temp[label] = temp[label] / sum

                e2lpd[example] = temp

            return e2lpd

    def get_workerquality(self):
        sum_worker = sum(self.weight.values())
        norm_worker_weight = dict()
        for worker in self.weight.keys():
            norm_worker_weight[worker] = self.weight[worker] / sum_worker
        return norm_worker_weight

    def Run(self, iterr):

        self.Init_truth()
        while iterr > 0:
            self.workers_weight_calculation()
            self.examples_truth_calculation()

            iterr -= 1

        return self.get_e2lpd(), self.get_workerquality()


def getaccuracy(truthfile, predict_truth, datatype):
    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e, ptruth in predict_truth.items():

        if e not in e2truth:
            continue

        count += 1

        if datatype == 'continuous':
            raise NotImplementedError('Continuous dataype not supported')
            # tcount = tcount + math.fabs(ptruth - float(e2truth[e]))
        else:
            if ptruth == e2truth[e]:
                tcount += 1

    if datatype == 'continuous':
        return tcount / count
    else:
        return tcount * 1.0 / count


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


def run_CRH(dataset):
    datatype = "categorical"
    distancetype = r'0/1 loss'

    datafile = "./data/" + dataset + "/label.csv"
    e2wl, w2el, label_set = gete2wlandw2el(datafile)

    t1 =time.time()
    e2lpd, weight = CRH(e2wl, w2el, label_set, datatype, distancetype).Run(10)

    w2w = open("./data/tilcc_features/" + dataset + "/workers_weight.txt", "w")
    for worker, reliability in weight.items():
        w2w.write(worker + "\t" + str(reliability) + "\n")
    w2w.close()
    t2 = time.time()
    return t2-t1
