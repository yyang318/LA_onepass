import math
import time
from copy import deepcopy


def norm(x):
    sum_ = 0
    for ele in x:
        sum_ += ele**2
    return math.sqrt(sum_)


def list_minus(x1, x2):
    return [ele1 - ele2 for ele1, ele2 in zip(x1,x2)]


def list_column_max_index(list_,col_ind):
    max_ind = -999
    max_value = -999
    for row_ind, row in enumerate(list_):
        if row[col_ind] > max_value:
            max_value = row[col_ind]
            max_ind = row_ind
    return max_ind


def list_list_sum(l1, l2):
    return [ele1 + ele2 for ele1, ele2 in zip(l1,l2)]


def find_centroid(ins, i):
    index = list_column_max_index(ins,i)
    return ins[index]


def list_all_true(list_):
    return False if False in list_ else True


def lists_elementwise_equal(l1,l2):
    return [ele1 == ele2 for ele1, ele2 in zip(l1,l2)]


def kmeans(dataset, feature):
    with open("./data/tilcc_features/" + dataset + "/" + feature + ".txt", "r") as f:
        instances = f.readlines()
    thetas = []
    f_len = len(instances[0].split("\t"))
    for instance in instances:
        theta = []
        for i in range(f_len):
            theta.append(float(instance.split("\t")[i].strip()))
        thetas.append(theta)

    t1 = time.time()
    # set Centroid
    ins = thetas

    centroid = []
    k = f_len - 2
    for i in range(k):
        centroid.append(find_centroid(ins, i))

    max_iti = 100
    iti = 0
    final_cluster = []
    while iti < max_iti:
        iti = iti + 1
        cluster = []
        for i in range(k):
            cluster.append([])

        for i in range(len(ins)):
            min_distance = norm(list_minus(centroid[0][0:-1], ins[i][0:-1]))
            min_index = 0
            for j in range(1, k):
                eucl = norm(list_minus(centroid[j][0:-1], ins[i][0:-1]))

                if eucl < min_distance:
                    min_distance = eucl
                    min_index = j
            cluster[min_index].append(ins[i])

        # update centroid
        temp_centroid = []
        for i in range(k):
            sum_j = cluster[i][0]
            for j in range(1, len(cluster[i])):
                sum_j = list_list_sum(sum_j,cluster[i][j])

            mean = [ele/len(cluster[i]) for ele in sum_j]

            min_distance = norm(list_minus(mean[0:-1], cluster[i][0][0:-1]))
            min_index = 0
            for j in range(1, len(cluster[i])):
                eucl = norm(list_minus(mean[0:-1],  cluster[i][j][0:-1]))
                if eucl < min_distance:
                    min_distance = eucl
                    min_index = j

            temp_centroid.append(cluster[i][min_index])

        # the condition of the convergence
        is_convergence = 0
        for i in range(k):
            if not list_all_true(lists_elementwise_equal(centroid[i],temp_centroid[i])):
                is_convergence = 1
        if is_convergence == 1:
            centroid = deepcopy(temp_centroid)
        else:
            final_cluster = cluster
            break
    t2 = time.time()

    with open("./data/" + dataset + "/truth.txt", "r") as f:
        truthlines = f.readlines()

    pred = []
    true = []

    truth_dict = dict()
    for line in truthlines:
        questionId = int(line.split("\t")[0])
        value = int(line.split("\t")[1].strip())
        truth_dict[questionId] = value

    common = 0
    trueNum = 0
    for i in range(k):
        for j in range(len(final_cluster[i])):
            questionId = int(final_cluster[i][j][k + 1])
            if questionId in truth_dict:
                common = common + 1
                pred.append(truth_dict[questionId])
                true.append(i + 1)
                if truth_dict[questionId] == i + 1:
                    trueNum = trueNum + 1

    return trueNum / common, t2 - t1, iti + 1