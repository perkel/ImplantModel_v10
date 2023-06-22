#  plot_inverse_results.py
#  This script takes the latest (from common_params.py) run and plots the results

import matplotlib.pyplot as plt
from common_params import *  # import common values across all models
import subject_data


def plot_inverse_results(use_fwd_model, txt_string, unsupervised):
    # Key constants to set
    save_fig = True

    if use_fwd_model:
        print('error -- this is for subject fits')
        scenario = txt_string
    else:
        subject = txt_string[0]
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
    figcols = 2
    fig_consol, axs = plt.subplots(figrows, figcols)
    fig_consol.set_figheight(6)
    fig_consol.set_figwidth(9)
    axs[0, 0].plot(xvals+0.1, thrsim[0][0], marker='o', color='lightblue', label='fit MP')
    axs[0, 0].plot(xvals-0.1, thrtargs[0][0], marker='o', color='blue', label='measured MP')
    axs[0, 0].plot(xvals[1:l_e]+0.1, thrsim[1][0][1:l_e], marker='o', color='pink', label='fit TP')
    axs[0, 0].plot(xvals[1:l_e]-0.1, thrtargs[1][0][1:l_e], marker='o', color='red', label='measured TP')
    yl = 'Threshold (dB)'
    mean_thr_err = (np.nanmean(np.abs(np.array(thrsim[0])-np.array(thrtargs[0]))) +
                    np.nanmean(np.abs(np.array(thrsim[1])-np.array(thrtargs[1]))))/2.0
    if use_fwd_model:
        title_text = 'Known scenario thresholds: ' + scenario + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    else:
        title_text = 'Subject thresholds: ' + subject + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    axs[0, 0].set(ylabel=yl)
    axs[0, 0].set_xlim(0, 17)
    axs[0, 0].set_ylim(33, 67)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticks([35, 45, 55, 65])
    axs[0, 0].legend(loc='upper right', ncol=2)

    title_text = 'Fit and actual positions; mean position error (mm): ' + '%.2f' % rpos_err_metric
    axs[1, 0].plot(xvals[1:l_e]+0.1, 1 - fitrposvals[1:l_e], marker='o', color='gray', label='fit')
    if use_fwd_model:
        axs[1, 0].plot(xvals[1:l_e]-0.1, 1 - rposvals[1:l_e], marker='o', color='black', label='actual')
    else:
        if np.any(ct_vals):
            axs[1, 0].plot(xvals, 1 - ct_vals-0.1, marker='o', color='black', label='CT estimate')

    # axs[1].plot(xvals[1:l_e], initvec[1:l_e], marker='o', color='blue', label='start')

    axs[1, 0].set(ylabel='Electrode distance (mm)')
    axs[1, 0].set_xlim(0, 17)
    axs[1, 0].set_ylim(0, 2)
    axs[1, 0].set_xticklabels([])
    axs[1, 0].legend()

    title_text = 'Fit survival values'
    if use_fwd_model:
        axs[2, 0].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='red', label='fit')
        axs[2, 0].plot(xvals[1:l_e], survvals[1:l_e], marker='o', color='green', label='desired')
    else:
        axs[2, 0].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='gray', label='modeled')

    axs[2, 0].set(xlabel='Electrode number', ylabel='Fractional neuronal density')
    axs[2, 0].set_xlim(0, 17)
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].legend()


# Now do second subject
    subject = txt_string[1]
    data_filename = INVOUTPUTDIR + subject + '_fitResults_' + 'combined.npy'
    [sigma_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
     rposerrs, rpos_err_metric, survivalerrs, ct_vals] = np.load(data_filename, allow_pickle=True)
    axs[0, 1].plot(xvals+0.1, thrsim[0][0], marker='o', color='lightblue', label='fit MP')
    axs[0, 1].plot(xvals-0.1, thrtargs[0][0], marker='o', color='blue', label='measured MP')
    axs[0, 1].plot(xvals[1:l_e]+0.1, thrsim[1][0][1:l_e], marker='o', color='pink', label='fit TP')
    axs[0, 1].plot(xvals[1:l_e]-0.1, thrtargs[1][0][1:l_e], marker='o', color='red', label='measured TP')
    yl = 'Threshold (dB)'
    mean_thr_err = (np.nanmean(np.abs(np.array(thrsim[0])-np.array(thrtargs[0]))) +
                    np.nanmean(np.abs(np.array(thrsim[1])-np.array(thrtargs[1]))))/2.0
    if use_fwd_model:
        title_text = 'Known scenario thresholds: ' + scenario + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    else:
        title_text = 'Subject thresholds: ' + subject + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    # axs[0, 1].set(xlabel='Electrode number', ylabel=yl)
    axs[0, 1].set_xlim(0, 17)
    axs[0, 1].set_ylim(33, 67)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].set_yticks([35, 45, 55, 65])

    axs[0, 1].legend(loc='upper right', ncol=2)

    title_text = 'Fit and actual positions; mean position error (mm): ' + '%.2f' % rpos_err_metric
    axs[1, 1].plot(xvals[1:l_e]+0.1, 1 - fitrposvals[1:l_e], marker='o', color='gray', label='fit')
    if use_fwd_model:
        axs[1, 1].plot(xvals[1:l_e]-0.1, 1 - rposvals[1:l_e], marker='o', color='black', label='actual')
    else:
        if np.any(ct_vals):
            axs[1, 1].plot(xvals, 1 - ct_vals-0.1, marker='o', color='black', label='CT estimate')

    # axs[1].plot(xvals[1:l_e], initvec[1:l_e], marker='o', color='blue', label='start')

    # axs[1, 1].set(xlabel='Electrode number', ylabel='Electrode distance (mm)')
    axs[1, 1].set_xlim(0, 17)
    axs[1, 1].set_ylim(0, 2)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].legend()

    title_text = 'Fit survival values'
    if use_fwd_model:
        axs[2, 1].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='red', label='fit')
        axs[2, 1].plot(xvals[1:l_e], survvals[1:l_e], marker='o', color='green', label='desired')
    else:
        axs[2, 1].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='gray', label='modeled')

    axs[2, 1].set(xlabel='Electrode number')
    axs[2, 1].set_xlim(0, 17)
    axs[2, 1].set_ylim(0, 1)
    axs[2, 1].set_yticklabels([])
    axs[2, 1].legend()

    fig_consol.tight_layout()

    if save_fig:
        save_file_name = INVOUTPUTDIR + txt_string[0] + txt_string[1] + '_fitResultsFig_' + 'combined.png'
        fig_consol.savefig(save_file_name)
        save_file_name = INVOUTPUTDIR + txt_string[0] + txt_string[1] + '_fitResultsFig_' + 'combined.eps'
        fig_consol.savefig(save_file_name, format='eps')



    if not unsupervised:
        plt.show()


if __name__ == '__main__':
    use_fwd_model = False
    txt_string = ['S40', 'S42']  # 2 subjects to fit side by side Fig 7 of the paper
    txt_string = ['S29', 'S56']  # 2 subjects to fit side by side Fig 8 of the paper
    unsupervised = False
    plot_inverse_results(use_fwd_model, txt_string, unsupervised)
