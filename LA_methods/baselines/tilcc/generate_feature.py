import csv
import time

# import numpy as np
from utils import list_sum

def generate_CRH_feature(dataset):
    # csv_file = csv.reader(open("./data/" + dataset + "/answer.csv", "r"))
    csv_file = csv.reader(open("./data/" + dataset + "/label.csv", "r"))

    qid = dict()
    qid_value = 1
    aid = dict()
    aid_value = 1
    instances = []
    for line in csv_file:
        # if line[0] != "question":
        if line[0] != "item":
            question = line[0]
            worker = line[1]
            answer = line[2]
            if question not in qid:
                qid[question] = qid_value
                qid_value += 1
            if answer not in aid:
                aid[answer] = aid_value
                aid_value += 1
            instances.append([worker, qid[question], aid[answer]])

    q2wl = dict()
    for instance in instances:
        questionId = instance[1]
        if questionId not in q2wl:
            q2wl[questionId] = []
        q2wl[questionId].append(instance)

    t1 = time.time()
    with open("./data/tilcc_features/" + dataset + "/workers_weight.txt", "r") as f:
        weights = f.readlines()

    w2w = dict()
    for line in weights:
        worker = line.split("\t")[0]
        weight = float(line.split("\t")[1].strip())
        w2w[worker] = weight

    # thetas = np.zeros(shape=(len(q2wl), len(aid) + 1), dtype=float)
    thetas = []
    for _ in range(len(q2wl)):
        thetas.append([0 for _ in range(len(aid)+1) ])
    for i in range(1, len(q2wl) + 1):
        ins = q2wl[i]
        # answers = np.zeros(len(aid))
        answers = [0 for _ in range(len(aid))]
        for j in range(len(ins)):
            answers[ins[j][2] - 1] = answers[ins[j][2] - 1] + w2w[ins[j][0]]
        for k in range(len(aid)):
            # thetas[i - 1][k] = answers[k] / answers.sum()
            thetas[i - 1][k] = answers[k] / list_sum(answers)
        theta_z = 0
        for k in range(1, len(aid)):
            theta_z = theta_z + thetas[i - 1][k] - thetas[i - 1][k - 1]
        thetas[i - 1][len(aid)] = theta_z / len(aid)

    q_feature = open("./data/tilcc_features/" + dataset + "/feature.txt", "w")
    for i in range(1, len(q2wl) + 1):
        for k in range(len(aid) + 1):
            q_feature.write(str(thetas[i - 1][k]))
            q_feature.write("\t")
        q_feature.write(str(i))
        q_feature.write("\n")
    q_feature.close()

    t2 = time.time()

    truth = open("./data/" + dataset + "/truth.txt", "w")
    csv_gold = csv.reader(open("./data/" + dataset + "/truth.csv", "r"))

    for line in csv_gold:
        # if line[0] != "question":
        if line[0] != "item":
            question = line[0]
            answer = line[1]
            if question in qid:
                truth.write(str(qid[question]) + "\t" + str(aid[answer]) + "\n")
    truth.close()

    return t2-t1
