# Import critical packages
import matplotlib.pyplot as plt
from common_params import *
import pickle
import csv
from matplotlib.font_manager import findfont, FontProperties


def PlotNeuronActivation():
    font = findfont(FontProperties(family=['sans-serif']))
    print('font is ', font)
    ACTIVATION_FILE = FWDOUTPUTDIR + 'neuronact_' + STD_TEXT + '.npy'
    print("Activation file: ", ACTIVATION_FILE)
    [survVals, rposVals, neuronact] = np.load(ACTIVATION_FILE, allow_pickle=True)

    hires = '_hi_res.npy'
    descrip = "surv_" + str(np.min(survVals)) + "_" + str(np.max(survVals)) + "_rpos_" + \
              str(np.min(rposVals)) + "_" + str(np.max(rposVals)) + hires

    params_file = FWDOUTPUTDIR + 'simParams' + scenarios[0]
    # sp = np.load(params_file + '.npy', allow_pickle=True)
    with open(params_file + '.pickle', 'rb') as f:
        sp = pickle.load(f)

    # Sanity check: is the sum across neuronact values really 100?
    posvals = np.arange(0, 33, 0.01) - 14.6 - 3 * ESPACE

    # Load monopolar data
    datafile = FWDOUTPUTDIR + "Monopolar_2D_" + STD_TEXT + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        mono_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            mono_thr[i, :] = row

    # Load tripolar data
    datafile = FWDOUTPUTDIR + "Tripolar_09_2D_" + STD_TEXT + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        tripol_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            tripol_thr[i, :] = row

    # Make plots
    survidxs = []
    rposidxs = []
    plt_surv_vals = [0.8]  # desired survival values to be plotted
    nsurv = len(plt_surv_vals)
    plt_rpos_vals = [-0.5, 0.0, 0.5]  # desired rpos values to be plotted
    nrpos = len(plt_rpos_vals)
    fig, axs = plt.subplots(nrpos, nsurv, sharex=True, sharey=True, figsize=(5, 8))

    for i, val in enumerate(plt_surv_vals):
        theidx = np.argmin(np.abs(survVals - val))
        survidxs.append(theidx)

    for i, val in enumerate(plt_rpos_vals):
        theidx = np.argmin(np.abs(rposVals - val))
        rposidxs.append(theidx)

    # Set labels and plot
    for i, ax in enumerate(axs.flat):
        row = int(i / nsurv)
        col = int(np.mod(i, nsurv))
        if row == 0:
            titletext = 'surv = %.2f' % survVals[survidxs[col]]
            ax.set_title(titletext)

        n_p_c = sp['neurons']['neur_per_clust']
        ax.plot(posvals + 1.1, neuronact[survidxs[col], rposidxs[row], 0, 0, :]/n_p_c, '.',
                color='blue', linewidth=0.5)
        ax.plot(posvals + 1.1, neuronact[survidxs[col], rposidxs[row], 1, 0, :]/n_p_c, '.',
                color='red', linewidth=0.5)
        ax.set(xlabel='Longitudinal distance (mm)')
        if row == 1 and col == 0:
            ax.set(ylabel='Fractional neuronal activation')
        # place threshold values here
        xlimit = [-4, 4]
        ax.set_xlim((xlimit[0], xlimit[1]))
        ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.label_outer()
        if col == (nsurv - 1):
            mytext = 'dist = ' + str(1 - plt_rpos_vals[row])
            ax.text(2.5, 0.07, mytext, horizontalalignment='right')

        print('surv ', titletext, ' dist ', str(1 - plt_rpos_vals[row]), ' Thr: ', mono_thr[survidxs[col],
                        rposidxs[row]], ' and ', tripol_thr[survidxs[col], rposidxs[row]])
    # plt.show()

    # Different graph, more similar to the one I made manually in Prism
    # nrows = 3
    # fig2, axs2 = plt.subplots(nrows, nsurv, sharex=True, sharey=True)
    # for i, ax in enumerate(axs2.flat):
    #     row = np.int(i / nrows)
    #     col = int(np.mod(i, nsurv))
    #     if row == 0:
    #         titletext = 'surv = %.2f' % survVals[survidxs[col]]
    #         ax.set_title(titletext)
    #     ax.plot(posvals + 1.1, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b')
    #     ax.plot(posvals, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b--', linewidth=1)
    #     ax.plot(posvals + 2.2, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b--', linewidth=1)
    #
    #     ax.plot(posvals - 1.1, neuronact[survidxs[col], rposidxs[row], 1, 0, :], 'r')
    #     ax.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')
    #     ax.set_xlim((-5.2, 5.2))
    #     ax.label_outer()
    #
    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.plot(posvals + 1.1, neuronact[5, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[10, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[15, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[18, 30, 0, 0, :], 'b')
    # ax3.set_xlim((-3, 3))
    # ax3.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')

    # fig4, ax4 = plt.subplots(1, 1)
    # ax4.plot(posvals - 1.1, neuronact[5, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[10, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[15, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[18, 30, 1, 1, :], 'r')
    # ax4.set_xlim((-3, 3))
    # ax4.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')
    #
    # fig5, ax5 = plt.subplots(3, 2, sharex=True, sharey=True)
    # fig5.tight_layout()
    # ax5[0, 0].plot(posvals + 1.1, neuronact[10, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[20, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[30, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[40, 24, 0, 0, :], 'b')
    # ax5[0, 0].set_title('Monopolar')
    # ax5[0, 0].annotate('Pos = -0.5', (-2, 60))
    # ax5[0, 1].plot(posvals - 1.1, neuronact[10, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[20, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[30, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[40, 24, 1, 1, :], 'r')
    # ax5[0, 1].set_title('Tripolar')
    #
    # ax5[1, 0].plot(posvals + 1.1, neuronact[10, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[20, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[30, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[40, 48, 0, 0, :], 'b')
    # ax5[1, 0].set_ylabel('Neurons per cluster')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[10, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[20, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[30, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[40, 48, 1, 1, :], 'r')
    #
    # ax5[2, 0].plot(posvals + 1.1, neuronact[10, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[20, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[30, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[40, 72, 0, 0, :], 'b')
    # ax5[2, 0].annotate('Pos = 0.5', (-2, 60))
    # ax5[2, 0].set_xlabel('Distance from electrode (mm)')
    #
    # ax5[2, 1].plot(posvals - 1.1, neuronact[10, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[20, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[30, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[40, 72, 1, 1, :], 'r')
    # ax5[2, 1].set_xlabel('Distance from electrode (mm)')
    # ax5[0, 0].set_xlim((-3, 3))
    # ax5[0, 0].set_ylim((0, 80))

    # Save figure
    fig_filename = FWDOUTPUTDIR + 'fig_neuronact.eps'
    plt.savefig(fig_filename, format='eps')

    plt.show()




if __name__ == '__main__':
    PlotNeuronActivation()
