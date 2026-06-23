"""Genomic evaluation metrics: k-mer correlation, regional correlation."""

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

from MuRaL.utils.info_utils import parse_info


def f3mer_comp(data_and_prob):
    """Compare observed and predicted mutation frequencies in 3-mers."""
    obs_pred_freq = data_and_prob[['us1', 'ds1', 'mut_type', 'prob']].groupby(['us1', 'ds1']).mean()
    return obs_pred_freq['mut_type'].corr(obs_pred_freq['prob'])


def freq_kmer_comp_multi(data_and_prob, k, n_class, use_obs_count=False):
    """Compare observed and predicted mutation frequencies in k-mers.

    Returns a list of correlations, one per mutation class.
    """
    d = k // 2
    mer_list = ['us' + str(i) for i in list(range(1, d + 1))[::-1]] + \
               ['ds' + str(i) for i in list(range(1, d + 1))]
    prob_list = ['prob' + str(i) for i in range(n_class)]

    corr_list = []
    for i in range(n_class):
        if use_obs_count:
            obs_col = data_and_prob['sample_weight'].where(
                data_and_prob['mut_type'] == i, 0.0).rename('obs')
        else:
            obs_col = (data_and_prob['mut_type'] == i).astype(float).rename('obs')
        obs_pred_freq = pd.concat(
            [data_and_prob[mer_list + [prob_list[i]]], obs_col], axis=1)
        obs_pred_freq = obs_pred_freq.groupby(mer_list).mean()

        corr_list.append(
            obs_pred_freq['obs'].astype(float).corr(
                obs_pred_freq[prob_list[i]].astype(float)))
    return corr_list


def corr_calc_sub(data, window, prob_names, use_obs_count=False):
    """Calculate regional correlations for sliding windows.

    Returns a list of Pearson correlations, one per mutation class.
    """
    n_class = len(prob_names)
    obs = [0] * n_class
    pred = [0] * n_class
    count = 0
    n_sites = len(data)

    avg_names = []
    for i in range(n_class):
        avg_names = avg_names + ['avg_obs' + str(i), 'avg_pred' + str(i)]

    last_chrom = data.loc[0, 'chrom']
    last_start = data.loc[0, 'start'] // window * window

    result = pd.DataFrame(columns=avg_names)
    for i in range(n_sites):
        start = data.loc[i, 'start'] // window * window
        chrom = data.loc[i, 'chrom']

        if chrom != last_chrom or start != last_start:
            avg_list = []
            for j in range(n_class):
                avg_list += [obs[j] / count, pred[j] / count]
            result = result.append(pd.DataFrame([avg_list], columns=avg_names))
            obs = [0] * n_class
            pred = [0] * n_class
            count = 0
            last_chrom = chrom
            last_start = start

        if not use_obs_count:
            obs_count = 1
        else:
            obs_count = parse_info(data.loc[i, 'info'])
        obs[int(data.loc[i, 'mut_type'])] += obs_count

        for j in range(n_class):
            pred[j] += data.loc[i, prob_names[j]]
        count = count + 1

    avg_list = []
    for j in range(n_class):
        avg_list += [obs[j] / count, pred[j] / count]
    result = result.append(pd.DataFrame([avg_list], columns=avg_names))

    corr_list = []
    for i in range(n_class):
        if sum(list(result['avg_obs' + str(i)] == 0) |
               (result['avg_obs' + str(i)] == 1)) / result.shape[0] > 0.5:
            print('Warning: too many zeros/ones (>50%) in the obs windows of size',
                  window, 'subtype', i)
        CV_obs = result['avg_obs' + str(i)].std() / result['avg_obs' + str(i)].mean()
        CV_pred = result['avg_pred' + str(i)].std() / result['avg_pred' + str(i)].mean()
        print('CV for ', str(window) + 'bp:', CV_obs, CV_pred)

        if result.shape[0] >= 3:
            corr = pearsonr(result['avg_obs' + str(i)],
                            result['avg_pred' + str(i)])[0]
        else:
            corr = 0
            print('Warning: too few windows for calculating correlation',
                  window, 'subtype', i)
        corr_list.append(corr)
    return corr_list


def calc_avg_prob(df, n_class, use_obs_count=False):
    """Calculate average observed and predicted probabilities per class."""
    avg_list = []
    if use_obs_count:
        total_count = df['sample_weight'].sum()
        for i in range(n_class):
            avg_list.append(
                df['sample_weight'].where(df['mut_type'] == i, 0.0).sum() / total_count)
    else:
        for i in range(n_class):
            avg_list.append(sum(list(df['mut_type'] == i)) / df.shape[0])
    for i in range(n_class):
        avg_list.append(df['prob' + str(i)].mean())
    return avg_list
