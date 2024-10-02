import torch, time, random, sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy


    
def calculate_spread_skill(ytrue, ymean, ystd, model_rmse,
                           bins=None, nBins=20,
                           returnBins=False):
    
    # taken from https://github.com/thunderhoser/cira_uq4ml/blob/main/regression_multi_datasets.ipynb
    # here we assume that ytrue etc correspond each to ONE variable (wind, pres, lat, lon)
    
    minBin = np.min([model_rmse, ystd.min()])
    maxBin = np.ceil(np.max([model_rmse, ystd.max()]))
    if not bins:
        bins = np.round(np.linspace(minBin, maxBin, nBins + 1), 1)
    else:
        nBins = len(bins) - 1

    error = np.empty((nBins))
    spread = np.empty((nBins))
    for i in range(nBins):
        refs = np.logical_and(ystd >= bins[i], ystd < bins[i + 1])
        if np.count_nonzero(refs) > 0:
            ytrueBin = ytrue[refs]
            ymeanBin = ymean[refs]
            error[i] = np.sqrt(np.mean((ytrueBin - ymeanBin)**2, axis=0))
            spread[i] = np.mean(ystd[refs])
            if error[i] > maxBin: # not sure why there is a bug for longitude, preventive fix
                error[i] = -999
                spread[i] = -999
        else:
            error[i] = -999
            spread[i] = -999
    
    if returnBins:
        return spread, error, bins
    else:
        return spread, error
    
    
### Taken from https://github.com/thunderhoser/cira_uq4ml/blob/main/regression_multi_datasets.ipynb ###

def get_histogram(var, bins=10, density=False, weights=None):
    counts, bin_edges = np.histogram(
        var, bins=bins, density=density, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers

def get_pit_dvalue(pit_counts):
    dvalue = 0.
    nbins = pit_counts.shape[0]
    nbinsI = 1./nbins

    pitTot = np.sum(pit_counts)
    pit_freq = np.divide(pit_counts, pitTot)
    for i in range(nbins):
        dvalue += (pit_freq[i] - nbinsI) * (pit_freq[i] - nbinsI)
    dvalue = np.sqrt(dvalue/nbins)
    return dvalue


def get_pit_evalue(nbins, tsamples):
    evalue = (1 - 1/nbins)/(tsamples * nbins)
    return np.sqrt(evalue)


def get_pit_points(y_true, y_mean, y_std,
                   pit_bins=None):

    if pit_bins is None:
      pit_bins = list(np.arange(0, 1.05, 0.05))

    nbins = len(pit_bins)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_mean = y_mean.reshape(-1, 1)
        y_std = y_std.reshape(-1, 1)
        
    nSamples = y_true.shape[0]
    nTargets = y_true.shape[-1]
    pDict = {}

    if nTargets > 1:
        pit_centers_all = []
        pit_counts_all = []
        pit_dvalues_all = []
        pit_values_all = []
        pit_weights_all = []
    for tg in range(nTargets):
        pit_values = scipy.stats.norm.cdf(x=y_true[..., tg],
                              loc=y_mean[..., tg],
                              scale=y_std[..., tg]).reshape(-1)
        weights = np.ones_like(pit_values) / pit_values.shape[0]
        pit_counts, bin_centers = get_histogram(pit_values,
                                                bins=pit_bins,
                                                weights=weights)
        pit_dvalue = get_pit_dvalue(pit_counts)
        if nTargets > 1:
            pit_dvalues_all.append(pit_dvalue)
            pit_centers_all.append(bin_centers)
            pit_counts_all.append(pit_counts)
            pit_values_all.append(pit_values)
            pit_weights_all.append(weights)
        else:
            pit_dvalues_all = pit_dvalue
            pit_centers_all = bin_centers
            pit_counts_all = pit_counts
            pit_values_all = pit_values
            pit_weights_all = weights

    pDict['pit_centers'] = pit_centers_all
    pDict['pit_counts'] = pit_counts_all
    pDict['pit_dvalues'] = pit_dvalues_all
    pDict['pit_evalues'] = get_pit_evalue(nbins, nSamples)
    pDict['pit_values'] = pit_values_all
    pDict['pit_weights'] = pit_weights_all
    return pDict


def list_to_str(input_list, format="%.2f"):
    """ Use format="%d" for an integer"""
    return [format % i for i in input_list]


def plot_pit_dict(pDict,
                  bar_color=None,
                  bar_label=None,
                  font_size=14,
                  hline=0.1,
                  legend_show=True,
                  legend_loc='best',
                  legend_size=14,
                  save_file=True,
                  save_path="/users/lpoulain/louis/plots/cnn/Final_comp/PIT/",
                  model='',
                  ldt=6,
                  split='',
                  show_error=True,
                  title=''):
  
    if bar_color is None:
      bar_color = ['tab:orange', 'seagreen',
                   'tab:purple', 'deeppink',
                   'tab:olive', 'tab:cyan',
                   'tab:brown', 'tab:gray']

    expList = list(pDict.keys())
    nExps = len(expList)
    
    pit_centers = pDict[expList[0]]['pit_centers']
    pit_labels = list_to_str(pit_centers)
    nBins = len(pit_centers)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    barWidth = 0.8 / nExps
    r1 = np.arange(nBins) - barWidth * 0.5 * nExps
    eval1 = np.ones(nBins) / 10.

    iExp = 0
    for exp in expList:
        eDict = pDict[exp]
        
        pit_counts = eDict['pit_counts']
        rNow = [x + barWidth * iExp for x in r1]

        if bar_label is not None:
            barLabel = bar_label[iExp]
            if show_error:
                try:
                    pit_dvalue = eDict['pit_dvalues']
                    barLabel += ' (D: {:.3f})'.format(pit_dvalue)
                    # if iExp == 0:
                    #    pit_evalue = eDict['pit_evalues']
                    #    barLabel += '(E: {:.3f})'.format(pit_evalue)
                except KeyError:
                    pass
        else:
            barLabel = None

        ax.bar(rNow, pit_counts,
               color=bar_color[iExp],
               edgecolor='black',
               label=barLabel,
               width=barWidth)
        iExp += 1

    if legend_show and bar_label is not None:
        ax.legend(fontsize=legend_size, loc=legend_loc)

    rfinal = np.arange(nBins) - barWidth / 2 * 0.25 * nExps
    if hline is not None:
        xlims = ax.get_xlim()
        ax.plot(xlims, [hline, hline],
                alpha=0.6,
                color='black',
                linestyle='--')
        ax.set_xlim(xlims)

    ax.set_xticks(rfinal, pit_labels)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Probability")
    ax.set_title(title, fontsize=font_size)
    if save_file:
        fig.savefig(save_path + f"pit_plot_{model}_{ldt}_{split}" + '.pdf', bbox_inches='tight')
    plt.show()

    plt.close()
    return

### ---------------------------------------------------------------------------------------------  ###