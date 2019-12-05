import numpy as np
from collections import defaultdict
from collections import OrderedDict
import heapq


def measure_k(probs, true_ys, k_list=[1, 3, 5]):
    max_k = np.max(k_list)
    num_samples = np.size(true_ys, 0)
    precision_k = defaultdict(float)
    dgg_k = defaultdict(float)
    ndgg_k = defaultdict(float)
    for i in range(num_samples):
        prob = probs[i, :]
        true_y = true_ys[i, :]
        prob = list(zip(prob, range(len(prob))))
        max_k_largest_index = [x[1] for x in heapq.nlargest(max_k, prob, key=lambda x: x[0])]
        for k in k_list:
            precision_k[k] += np.sum(true_y[max_k_largest_index[0:k]])/k
            dgg_k[k] += np.sum(true_y[max_k_largest_index[0:k]] / np.log2(2+np.arange(k)))
    for k in k_list:
        precision_k[k] /= num_samples
        dgg_k[k] /= num_samples
        ndgg_k[k] = dgg_k[k] / np.sum(1/np.log2(2+np.arange(k)))
    return precision_k, dgg_k, ndgg_k


def measure_b(pred_b, y):
    epsilon = 1e-9
    #micro
    tp = np.sum(np.logical_and(pred_b, y))
    fp = np.sum(np.logical_and(pred_b, np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(pred_b), y))
    micro_p = tp/(tp+fp+epsilon)
    micro_r = tp/(tp+fn+epsilon)
    micor_f1 = 2*micro_p*micro_r/(micro_p+micro_r)
    #marco
    tp = np.sum(np.logical_and(pred_b, y), 0)
    fp = np.sum(np.logical_and(pred_b, np.logical_not(y)), 0)
    fn = np.sum(np.logical_and(np.logical_not(pred_b), y), 0)
    marco_p = np.mean(tp/(tp+fp+epsilon))
    marco_r = np.mean(tp/(tp+fn+epsilon))
    marco_f1 = 2*marco_p*marco_r/(marco_p+marco_r)
    #Example based measures
    hamming_loss = np.mean(np.logical_xor(pred_b, y))
    accuracy = np.mean(np.sum(np.logical_and(pred_b, y), 1)/np.sum(np.logical_or(pred_b, y), 1))
    precision = np.mean(np.sum(np.logical_and(pred_b, y), 1)/np.sum(pred_b, 1))
    recall = np.mean(np.sum(np.logical_and(pred_b, y), 1)/(np.sum(y, 1)+epsilon))
    F1 = np.mean(2*np.sum(np.logical_and(pred_b, y), 1)/(np.sum(pred_b, 1)+np.sum(y, 1)))
    return micro_p, micro_r, micor_f1, marco_p, marco_r, marco_f1, hamming_loss, accuracy, precision, recall, F1


def measure_multi_label(probabilities, true_ys, mode, k_list=[1, 3, 5]):
    #measure_k
    result = OrderedDict()
    precision_k, dgg_k, ndgg_k = measure_k(probabilities, true_ys, k_list)
    for k in k_list:
        result[mode+'_Precision@{0:d}'.format(k)] = precision_k[k]
        result[mode+'_dgg@{0:d}'.format(k)] = dgg_k[k]
        result[mode+'_ndgg@{0:d}'.format(k)] = ndgg_k[k]
    # measure_b
    binaries = probabilities >= 0.5  # thresholds
    # revise to guarantee at least one is one
    index = np.argmax(probabilities, 1)
    binaries = np.reshape(np.array(binaries), [-1])
    index = index + np.size(probabilities, 1) * np.arange(np.size(probabilities, 0))
    binaries[index] = 1
    binaries.shape = np.size(probabilities, 0), np.size(probabilities, 1)
    #        print(np.sum(np.any(binaries,1)))
    binaries = binaries.astype(int)
    micro_p, micro_r, micor_f1, marco_p, marco_r, marco_f1, hamming_loss, accuracy, precision, recall, F1 = measure_b(
        binaries, true_ys)
    # print(
    #     'micro_p {0:.4f}\nmicro_r {1:.4f}\nmicor_f1 {2:.4f}\nmarco_p {3:.4f}\nmarco_r {4:.4f}\nmarco_f1 {5:.4f}\nhamming_loss {6:.4f}\naccuracy {7:.4f}\nprecision {8:.4f}\nrecall {9:.4f}\nF1 {10:.4f}\n'
    #         .format(micro_p, micro_r, micor_f1, marco_p, marco_r, marco_f1, hamming_loss, accuracy,
    #                 precision, recall, F1)
    # )
    result[mode+'_micro_p'] = micro_p
    result[mode+'_micro_r'] = micro_r
    result[mode+'_micor_f1'] = micor_f1
    result[mode+'_marco_p'] = marco_p
    result[mode+'_marco_r'] = marco_r
    result[mode+'_marco_f1'] = marco_f1
    result[mode+'_hamming_loss'] = hamming_loss
    result[mode+'_accuracy'] = accuracy
    result[mode+'_precision'] = precision
    result[mode+'_recall'] = recall
    result[mode+'_F1'] = F1
    return result


def measure_ex(binaries, true_ys):
    epsilon = 1e-9
    precision = np.sum(np.logical_and(binaries, true_ys), 1)/np.sum(binaries, 1)
    recall = np.sum(np.logical_and(binaries, true_ys), 1)/(np.sum(true_ys, 1)+epsilon)
    F1 = 2*np.sum(np.logical_and(binaries, true_ys), 1)/(np.sum(binaries, 1)+np.sum(true_ys, 1))
    return F1
