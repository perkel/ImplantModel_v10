#  plot_inverse_results.py
#  This script takes the latest (from common_params.py) run and plots the results

import matplotlib.pyplot as plt
from common_params import *  # import common values across all models
import subject_data


def plot_inverse_results(use_fwd_model, txt_string, unsupervised):
    # Key constants to set
    save_fig = True

    if use_fwd_model:
        scenario = txt_string
    else:
        subject = txt_string
        ct_vals = subject_data.subj_ct_data(subject)

    # Open file and load data
    if use_fwd_model:
        data_filename = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined.npy'
        [sigma_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
         rposerrs, rpos_err_metric, survivalerrs] = np.load(data_filename, allow_pickle=True)
    else:
        data_filename = INVOUTPUTDIR + subject + '_fitResults_' + 'combined.npy'
        [sigma_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
         rposerrs, rpos_err_metric, survivalerrs, ct_vals] = np.load(data_filename, allow_pickle=True)

    # Make plots
    xvals = np.arange(0, NELEC) + 1
    l_e = NELEC - 1  # last electrode to plot

    # All on one plot
    figrows = 3
    figcols = 1
    fig_consol, axs = plt.subplots(figrows, figcols)
    fig_consol.set_figheight(9)
    fig_consol.set_figwidth(7.5)
    axs[0].plot(xvals+0.1, thrsim[0][0], marker='o', color='lightblue', label='fit MP')
    axs[0].plot(xvals-0.1, thrtargs[0][0], marker='o', color='blue', label='measured MP')
    axs[0].plot(xvals[1:l_e]+0.1, thrsim[1][0][1:l_e], marker='o', color='pink', label='fit TP')
    axs[0].plot(xvals[1:l_e]-0.1, thrtargs[1][0][1:l_e], marker='o', color='red', label='measured TP')
    yl = 'Threshold (dB)'
    mean_thr_err = (np.nanmean(np.abs(np.array(thrsim[0])-np.array(thrtargs[0]))) +
                    np.nanmean(np.abs(np.array(thrsim[1])-np.array(thrtargs[1]))))/2.0
    if use_fwd_model:
        title_text = 'Known scenario thresholds: ' + scenario + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    else:
        title_text = 'Subject thresholds: ' + subject + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    axs[0].set(xlabel='Electrode number', ylabel=yl, title=title_text)
    axs[0].set_xlim(0, 17)
    axs[0].legend(loc='upper right', ncol=2)

    title_text = 'Fit and actual positions; mean position error (mm): ' + '%.2f' % rpos_err_metric
    axs[1].plot(xvals[1:l_e]+0.1, 1 - fitrposvals[1:l_e], marker='o', color='gray', label='fit')
    if use_fwd_model:
        axs[1].plot(xvals[1:l_e]-0.1, 1 - rposvals[1:l_e], marker='o', color='black', label='actual')
    else:
        if np.any(ct_vals):
            axs[1].plot(xvals, 1 - ct_vals-0.1, marker='o', color='black', label='CT estimate')

    # axs[1].plot(xvals[1:l_e], initvec[1:l_e], marker='o', color='blue', label='start')

    axs[1].set(xlabel='Electrode number', ylabel='Electrode distance (mm)', title=title_text)
    axs[1].set_xlim(0, 17)
    axs[1].set_ylim(0, 2)
    axs[1].legend()

    title_text = 'Fit survival values'
    if use_fwd_model:
        axs[2].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='red', label='fit')
        axs[2].plot(xvals[1:l_e], survvals[1:l_e], marker='o', color='green', label='desired')
    else:
        axs[2].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='black', label='modeled')

    axs[2].set(xlabel='Electrode number', ylabel='Fractional neuronal density', title=title_text)
    axs[2].set_xlim(0, 17)
    axs[2].set_ylim(0, 1)
    axs[2].legend()
    fig_consol.tight_layout()

    # -- could add plots of error (difference between desired/measured and fitted values)

    if save_fig:
        save_file_name = INVOUTPUTDIR + txt_string + '_fitResultsFig_' + 'combined.png'
        fig_consol.savefig(save_file_name)
        save_file_name = INVOUTPUTDIR + txt_string + '_fitResultsFig_' + 'combined.eps'
        fig_consol.savefig(save_file_name, format='eps')


    # test correlation figure
    if not use_fwd_model:
        fig2, ax2 = plt.subplots()
        ct_dist = 1-ct_vals[1:l_e]
        fit_dist = 1 - fitrposvals[1:l_e]
        fig2 = plt.plot(ct_dist, fit_dist, 'o')
        ax2.set(xlabel='CT distance (mm)', ylabel='fit distance (mm)')
        ax2.set_xlim([0, 2])
        ax2.set_ylim([0, 2])
        fit_corr = np.corrcoef(ct_dist, fit_dist)
    if not unsupervised:
        plt.show()


if __name__ == '__main__':
    use_fwd_model = True
    # txt_string = scenarios[0]
    #txt_string = 'S40'
    txt_string = 'RampRposSGradual80'
    unsupervised = False
    plot_inverse_results(use_fwd_model, txt_string, unsupervised)
