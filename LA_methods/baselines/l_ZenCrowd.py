import random
import time
from results_summary import OfflineResultsSummary
from data_pipeline import gete2wlandw2el, getaccuracy


class EM:

    def __init__(self, e2wl, w2el, label_set):

        self.e2wl = e2wl
        self.w2el = w2el
        self.label_set = label_set

        ###################################################################
        # Expectation Maximization
        ###################################################################
    def InitPj(self):
        l2pd={}
        for label in self.label_set:
            l2pd[label]=1.0/len(self.label_set)
        return l2pd


    def InitWM(self, workers):
        wm={}

        if workers=={}:
            workers=list(self.w2el.keys())
            for worker in workers:
                wm[worker]=0.8
        else:
            for worker in workers:
                if worker not in wm: # workers --> wm
                    wm[worker] = 0.8
                else:
                    wm[worker]=workers[worker]

        return wm



    #E-step
    def ComputeTij(self, e2wl, l2pd, wm):
        e2lpd={}
        for e, workerlabels in list(e2wl.items()):
            e2lpd[e]={}
            for label in self.label_set:
                e2lpd[e][label]=1.0#l2pd[label]

            for worker,label in workerlabels:
                for candlabel in self.label_set:
                    if label==candlabel:
                        e2lpd[e][candlabel]*=wm[worker]
                    else:
                        e2lpd[e][candlabel]*=(1-wm[worker])*1.0/(len(self.label_set)-1)

            sums=0
            for label in self.label_set:
                sums+=e2lpd[e][label]

            if sums==0:
                for label in self.label_set:
                    e2lpd[e][label]=1.0/len(self.label_set)
            else:
                for label in self.label_set:
                    e2lpd[e][label]=e2lpd[e][label]*1.0/sums

        #print e2lpd
        return e2lpd


    #M-step
    def ComputePj(self, e2lpd):
        l2pd = {}

        for label in self.label_set:
            l2pd[label]=0
        for e in e2lpd:
            for label in e2lpd[e]:
                l2pd[label]+=e2lpd[e][label]

        for label in self.label_set:
            l2pd[label]=l2pd[label]*1.0/len(e2lpd)

        return l2pd


    def ComputeWM(self, w2el, e2lpd):
        wm={}
        for worker,examplelabels in list(w2el.items()):
            wm[worker]=0.0
            for e,label in examplelabels:
                wm[worker]+=e2lpd[e][label]*1.0/len(examplelabels)

        return wm


    def Run(self, iter = 20, workers={}):
        #wm     worker_to_confusion_matrix = {}
        #e2pd   example_to_softlabel = {}
        #l2pd   label_to_priority_probability = {}


        l2pd = self.InitPj()
        wm = self.InitWM(workers)
        while iter>0:
            #E-step
            e2lpd = self.ComputeTij(self.e2wl, {}, wm)

            #M-step
            #l2pd = self.ComputePj(e2lpd)
            wm = self.ComputeWM(self.w2el, e2lpd)

            #print l2pd,wm

            iter -= 1

        return e2lpd, wm

def from_z_to_truths_multi(z_i):
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


    