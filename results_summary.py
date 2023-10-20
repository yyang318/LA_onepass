import math
from collections import defaultdict
import pandas as pd

from utils import list_mean,list_std,list_mean_2d,list_std_2d
import sys

class OfflineResultsSummary:
    def __init__(self,method):
        # super(ResultsSummary, self).__int__()
        self.method = method
        self.accuracy_results = {}
        self.runtime_results = {}
        self.accuracy_results = defaultdict(lambda :[],self.accuracy_results)
        self.runtime_results = defaultdict(lambda :[], self.runtime_results)
        self.iterations = {}
        self.iterations = defaultdict(lambda :[],self.iterations)
        self.elbos = {}
        self.elbos = defaultdict(lambda :[], self.elbos)

    def add(self,dataset, accuracy, runtime, iteration=1, elbo = None):
        self.accuracy_results[dataset].append(accuracy)
        self.runtime_results[dataset].append(runtime)
        self.iterations[dataset].append(iteration)
        if elbo is not None:
            self.elbos[dataset].append(elbo)


    def get_accuracy_list(self, dataset):
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        return self.runtime_results[dataset]

    def get_iteration_list(self, dataset):
        return self.iterations[dataset]

    def get_iteration_mean(self, dataset):
        return list_mean(self.iterations[dataset])

    def get_iteration_std(self, dataset):
        return list_std(self.iterations[dataset])

    def get_num_rounds(self, dataset):
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        return list_mean(self.accuracy_results[dataset])

    def get_accuracy_std(self, dataset):
        return list_std(self.accuracy_results[dataset])

    def get_runtime_mean(self, dataset):
        return list_mean(self.runtime_results[dataset])

    def get_runtime_std(self, dataset):
        return list_std(self.runtime_results[dataset])

    def get_overall_accuracy_mean(self):
        sum_ = 0
        for dataset, list_ in self.accuracy_results.items():
            sum_ += list_mean(list_)
        return sum_/len(self.accuracy_results)

    def get_overall_runtime_mean(self):
        sum_ = 0
        for dataset, list_ in self.runtime_results.items():
            sum_ += list_mean(list_)
        return sum_/len(self.runtime_results)

    def get_method(self):
        return self.method


    def get_EBCC_max_accuracy_runtime_iteration(self,dataset):
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        elbos = self.elbos
        max_val, max_ind = -sys.float_info.max, -sys.float_info.max
        for ind,ele in enumerate(elbos[dataset]):
            if ele > max_val:
                max_val = ele
                max_ind = ind
        return self.accuracy_results[dataset][max_ind], self.runtime_results[dataset][max_ind],self.iterations[dataset][max_ind]

    def get_EBCC_dataframe(self):
        # df = pd.DataFrame()
        # df.columns=['dataset','accuracy','runtime','iteration','elbo']
        if self.method != 'EBCC':
            raise ValueError('Not EBCC results')
        datasets = []
        accuracy_results = []
        runtime_results = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for accuracy, runtime, iteration,elbo in zip(self.accuracy_results[dataset],
                                                         self.runtime_results[dataset],
                                                         self.iterations[dataset],
                                                         self.elbos[dataset]):
                # df.append({'accuracy':accuracy, 'runtime':runtime,
                #            'iteration':iteration, 'elbo':elbo},ignore_index=True)
                datasets.append(dataset)
                accuracy_results.append(accuracy)
                runtime_results.append(runtime)
                iterations.append(iteration)
                elbos.append(elbo)
        df = pd.DataFrame({'dataset':datasets,'accuracy':accuracy_results, 'runtime':runtime_results,
                           'iteration':iterations, 'elbo':elbos})
        return df




    def display(self, elbo = False):
        for dataset in self.accuracy_results.keys():
            num_rounds = self.get_num_rounds(dataset)
            acc_mean = round(self.get_accuracy_mean(dataset),4)
            acc_std = round(self.get_accuracy_std(dataset),4)
            runtime_mean = round(self.get_runtime_mean(dataset),4)
            runtime_std = round(self.get_runtime_std(dataset),4)

            iteration_mean = round(self.get_iteration_mean(dataset),4)
            iteration_std = round(self.get_iteration_std(dataset),4)

            message = f'Method {self.method}:Dataset {dataset} runs {num_rounds} rounds, ' \
                      f'accuracy mean {acc_mean}, accuracy std {acc_std},' \
                      f' runtime mean {runtime_mean}, runtime std {runtime_std},' \
                      f' iteration mean {iteration_mean}, iteration std {iteration_std}'
            print(message)

    def to_dataframe_mean(self):
        df = pd.DataFrame({'dataset':list(self.accuracy_results.keys()),
                            'accuracy mean':[round(self.get_accuracy_mean(dataset),4) for dataset in self.accuracy_results.keys()],
                           'accuracy std': [round(self.get_accuracy_std(dataset),4) for dataset in self.accuracy_results.keys()],
                           'runtime mean': [round(self.get_runtime_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'runtime std': [round(self.get_runtime_std(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'iteration mean': [round(self.get_iteration_mean(dataset), 4) for dataset in
                                            self.accuracy_results.keys()],
                           'iteration std': [round(self.get_iteration_std(dataset), 4) for dataset in
                                              self.accuracy_results.keys()],
                           })

        return df

    def to_dataframe_detail(self):
        datasets = []
        accuracies = []
        runtimes = []
        iterations = []
        elbos = []
        for dataset in self.accuracy_results.keys():
            for ind, acc in enumerate(self.accuracy_results[dataset]):
                datasets.append(dataset)
                accuracies.append(acc)
                runtimes.append(self.runtime_results[dataset][ind])
                iterations.append(self.iterations[dataset][ind])
                if len(self.elbos[dataset]) > 0:
                    elbos.append(self.elbos[dataset][ind])
        if len(elbos) == 0:
            df = pd.DataFrame({'dataset':datasets,
                               'accuracy':accuracies,
                               'runtime':runtimes,
                               'iteration':iterations})
        else:
            df = pd.DataFrame({'dataset': datasets,
                               'accuracy': accuracies,
                               'runtime': runtimes,
                               'iteration': iterations,
                               'elbo':elbos})
        return df


class OnlineResultsSummary:
    def __init__(self,method, num_chunks = 10):
        # super(ResultsSummary, self).__int__()
        self.method = method
        self.num_chunks = num_chunks
        self.accuracy_results = {}
        self.runtime_results = {}
        self.accuracy_results = defaultdict(lambda :[],self.accuracy_results)
        self.runtime_results = defaultdict(lambda :[], self.runtime_results)
        # self.iterations = {}
        # self.iterations = defaultdict(lambda :[],self.iterations)

    def add(self,dataset, accuracy_list, runtime_list):
        self.accuracy_results[dataset].append(accuracy_list)
        self.runtime_results[dataset].append(runtime_list)
        # self.iterations[dataset].append(iteration)

    def get_accuracy_list(self, dataset):
        return self.accuracy_results[dataset]

    def get_runtime_list(self, dataset):
        return self.runtime_results[dataset]

    # def get_iteration_list(self, dataset):
    #     return self.iterations[dataset]

    # def get_iteration_mean(self, dataset):
    #     return list_mean(self.iterations[dataset])
    #
    # def get_iteration_std(self, dataset):
    #     return list_std(self.iterations[dataset])

    def get_num_rounds(self, dataset):
        return len(self.accuracy_results[dataset])

    def get_accuracy_mean(self, dataset):
        # return list_mean(self.accuracy_results[dataset])
        return list_mean_2d(self.accuracy_results[dataset],axis=0)

    def get_accuracy_std(self, dataset):
        # return list_std(self.accuracy_results[dataset])
        return list_std_2d(self.accuracy_results[dataset], axis=0)

    def get_runtime_mean(self, dataset):
        # return list_mean(self.runtime_results[dataset])
        return list_mean_2d(self.runtime_results[dataset], axis=0)

    def get_runtime_std(self, dataset):
        # return list_std(self.runtime_results[dataset])
        return list_std_2d(self.runtime_results[dataset], axis=0)



    def get_method(self):
        return self.method

    def to_dataframe_accuracy_all_mean(self):
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_mean(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_mean(self):
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_mean(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_all_std(self):
        entries = []
        for dataset in self.accuracy_results.keys():
            datset_accuracy_mean = self.get_accuracy_std(dataset)
            entries.append(datset_accuracy_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.accuracy_results.keys()))
        return df

    def to_dataframe_runtime_all_std(self):
        entries = []
        for dataset in self.runtime_results.keys():
            datset_runtime_mean = self.get_runtime_std(dataset)
            entries.append(datset_runtime_mean)
        df = pd.DataFrame(data=entries, columns=list(range(1, 11)), index=list(self.runtime_results.keys()))
        return df

    def to_dataframe_accuracy_detail(self):

        accuracy_dict = self.accuracy_results
        num_chunks = 10
        entries = []
        for dataset in accuracy_dict.keys():
            for ind, acc in enumerate(accuracy_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df

    def to_dataframe_runtime_detail(self):

        runtime_dict = self.runtime_results
        num_chunks = 10
        entries = []
        for dataset in runtime_dict.keys():
            for ind, acc in enumerate(runtime_dict[dataset]):
                row = [dataset]
                row += acc
                entries.append(row)
        columns = ['dataset']
        for i in range(num_chunks):
            columns.append(str(i + 1))
        df = pd.DataFrame(data=entries, columns=columns)
        return df

