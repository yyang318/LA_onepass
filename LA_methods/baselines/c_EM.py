import random
import time
from results_summary import OfflineResultsSummary
from data_pipeline import gete2wlandw2el, getaccuracy

class EM:

    #The DS algorithm
    def __init__(self,e2wl,w2el,label_set, initquality=0.7):
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = list(self.w2el.keys())
        self.label_set = label_set
        self.initalquality = initquality
             


# E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in list(self.e2wl.items()):
            lpd = {}
            total_weight = 0

            for tlabel, prob in list(self.l2pd.items()):
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]
                
                lpd[tlabel] = weight
                total_weight += weight
                

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution 
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight
            
            self.e2lpd[example] = lpd



#M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in list(self.e2lpd.items()):
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0/len(self.e2lpd)


            
    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0


        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            
            for tlabel in self.label_set:

                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (1-self.initalquality)*1.0/(len(self.label_set)-1)

                    continue

                for example, label in self.w2el[w]:
                    
                        self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel]*1.0/w2lweights[w][tlabel]



        return self.w2cm
                    

    
    

                     

#initialization
    def Init_l2pd(self):
        #uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0/len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1-self.initalquality)/(len(self.label_set)-1)

        return w2cm

    def Run(self, iterr = 20):
        
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()
        

        while iterr > 0:
            # E-step
            self.Update_e2lpd() 

            # M-step
            self.Update_l2pd()
            self.Update_w2cm()

            iterr -= 1
        
        return self.e2lpd, self.w2cm


    # def computelikelihood(self):
        
    #     lh = 0

    #     for _, worker_label_set in list(self.e2wl.items()):
    #         temp = 0
    #         for tlabel, prior in list(self.l2pd.items()):
    #             inner = np.log(prior)
    #             for worker, label in worker_label_set:
    #                 inner += np.log(self.w2cm[worker][tlabel][label]
    #             temp += inner
            
    #         lh += math.log(temp)
        
    #     return lh




