import random
import matplotlib.pyplot as plt
from LA_methods.proposed.proposed_method import one_pass
from collections import defaultdict
from copy import deepcopy
import numpy as np

def simulation(true_w_range, M=20, T=1000, K=4):
    label_set = [str(i) for i in range(K)]
    true_w = {}
    for i in range(M):
        if isinstance(true_w_range, float):
            true_w[str(i)] = true_w_range
        else:
            true_w[str(i)] = random.uniform(true_w_range[0],true_w_range[1])
    ground_truths = {}
    for i in range(T):
        ground_truths[str(i)] = str(random.randint(0,K-1))


    e2wl, w2el = {},{}
    e2wl = defaultdict(lambda :[],e2wl)
    w2el = defaultdict(lambda :[],w2el)

    for t in range(T):
        truth_t = ground_truths[str(t)]
        for i in range(M):
            prob = random.random()
            if prob <= true_w[str(i)]:
                label_it = truth_t
            else:
                wrong_candidates = deepcopy(label_set)
                wrong_candidates.remove(truth_t)
                label_it = random.choice(wrong_candidates)

            e2wl[str(t)].append([str(i),label_it])
            w2el[str(i)].append([str(t),label_it])


    one_pass_truths, a, a_evo = one_pass(e2wl, w2el, label_set, alpha=3, beta=2,shuffle=False,record_a_evo=True)

    correct = 0
    for item in one_pass_truths.keys():
        if one_pass_truths[item] == ground_truths[item]:
            correct += 1
    print(correct/T)

    return true_w, a_evo


def plot_same_quality(true_w_range=0.6, M=20, T=1000, K=4):
    true_w, a_evo = simulation(true_w_range, M, T, K)
    a_evo_np = np.zeros((T,M))
    for t in range(T):
        for m in range(M):
            a_evo_np[t,m] = a_evo[str(t)][str(m)]
    plt.figure(figsize=(10, 3), dpi=200)
    plt.hlines(true_w_range, 0, T, colors='red', linewidth=2, label='true w = 0.6')
    for i in range(M):
        qualities = [ele[i] for ele in a_evo_np]
        if i < M - 1:
            plt.plot(np.arange(0, T), qualities[0:], color='g', linewidth=1.0, alpha=0.3)
        else:
            plt.plot(np.arange(0, T), qualities[0:], color='g', linewidth=0.3, label='estimated w')

    plt.plot(np.arange(1, T), true_w_range - 1 / np.sqrt(np.arange(1, T)), color='blue', label=r'bound $\epsilon = 1$')
    plt.plot(np.arange(1, T), true_w_range + 1 / np.sqrt(np.arange(1, T)), color='blue')

    plt.plot(np.arange(1, T), true_w_range - 2 / np.sqrt(np.arange(1, T)), color='orange',
             label=r'bound $\epsilon = 2$')
    plt.plot(np.arange(1, T), true_w_range + 2 / np.sqrt(np.arange(1, T)), color='orange')

    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.ylim((0.2, 1.0))
    plt.xlim((-10, T + 10))
    plt.legend(loc='upper right', fontsize=11, ncol=2)
    plt.savefig('simulation.png', bbox_inches='tight')
    plt.show()


def plot_different_quality(true_w_range=[0.4, 0.8], M=20, T=1000, K=3):
    true_w, a_evos = simulation(true_w_range, M, T, K)
    a_evo_np = np.zeros((T, M))
    for t in range(T):
        for m in range(M):
            a_evo_np[t, m] = a_evos[str(t)][str(m)]
    true_w_np = np.zeros(M)
    for i in range(M):
        true_w_np[i] = true_w[str(i)]


    fig, axs = plt.subplots(5, 4, figsize=(15, 20))
    row, col = 0, 0

    for i in range(M):
        axs[row, col].hlines(true_w_np[i], 0, T, colors='red', linewidth=1.0,
                             label=r'true w = ' + str(round(true_w_np[i], 2)))

        label = r'estimated w'
        axs[row, col].plot(np.arange(0, T), a_evo_np[:, i], color='g', linewidth=1.0, label=label)

        label = r'bound $\epsilon = 1$'
        axs[row, col].plot(np.arange(1, T), true_w_np[i] - 1 / np.sqrt(np.arange(1, T)), color='blue', label=label)
        axs[row, col].plot(np.arange(1, T), true_w_np[i] + 1 / np.sqrt(np.arange(1, T)), color='blue')

        label = r'bound $\epsilon = 2$'
        axs[row, col].plot(np.arange(1, T), true_w_np[i] - 2 / np.sqrt(np.arange(1, T)), color='orange', label=label)
        axs[row, col].plot(np.arange(1, T), true_w_np[i] + 2 / np.sqrt(np.arange(1, T)), color='orange')

        if col != 0:
            axs[row, col].axes.yaxis.set_ticklabels([])
        axs[row, col].set_ylim((0, 1))
        axs[row, col].tick_params(axis='both', which='major', labelsize=16)

        legend_loc = 'upper right' if true_w_np[i] <= 0.5 else 'lower right'

        axs[row, col].legend(loc=legend_loc)
        col += 1
        if col >= 4:
            row += 1
            col = 0

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.15)
    # plt.savefig('simulation_full.png', bbox_inches='tight')
    plt.show()


    fig, axs = plt.subplots(2, 1, figsize=(10, 6), dpi=200)

    for i in range(2):
        axs[i].hlines(true_w_np[i], 0, T, colors='red', linewidth=1.0, label=r'true w = ' + str(round(true_w_np[i], 2)))

        label = r'estimated w'
        axs[i].plot(np.arange(0, T), a_evo_np[:, i], color='g', linewidth=1.0, label=label)

        label = r'bound $\epsilon = 1$'
        axs[i].plot(np.arange(1, T), true_w_np[i] - 1 / np.sqrt(np.arange(1, T)), color='blue', label=label)
        axs[i].plot(np.arange(1, T), true_w_np[i] + 1 / np.sqrt(np.arange(1, T)), color='blue')

        label = r'bound $\epsilon = 2$'
        axs[i].plot(np.arange(1, T), true_w_np[i] - 2 / np.sqrt(np.arange(1, T)), color='orange', label=label)
        axs[i].plot(np.arange(1, T), true_w_np[i] + 2 / np.sqrt(np.arange(1, T)), color='orange')


        axs[i].set_ylim((0, 1))
        axs[i].tick_params(axis='both', which='major', labelsize=10)

        if true_w_np[i] < 0.5:
            axs[i].legend(ncol=2, loc='upper right', fontsize=11)
        else:
            axs[i].legend(ncol=2, loc='lower right', fontsize=11)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.15)
    # plt.savefig('simulation_2.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # plot_same_quality()
    plot_different_quality()
