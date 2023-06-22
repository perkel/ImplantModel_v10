# Script to make figure 3 (3D plot) for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from scipy import interpolate
import intersection as intsec
from common_params import *
import shapely as shap


def fig_2D_contour():
    # this is an option to search systematically for unique or multiple solutions for a given set of
    # monopolar and tripolar threshold values
    map_unique_solutions = False
    filled_contours = False  # Fill the contour plots? Otherwise make contour lines
    # Set default figure values
    # mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2

    # Declare variables
    surv_vals = np.arange(0.04, 0.97, 0.02)
    nsurv = len(surv_vals)
    rpos_vals = np.arange(-0.95, 0.96, 0.02)
    nrpos = len(rpos_vals)

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

    print("average monopolar threshold: ", np.nanmean(mono_thr))

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

    print('Data retrieved from directory: ', FWDOUTPUTDIR)

    # Measure min/max/mean differences between monopolar and tripolar
    thr_diff = tripol_thr - mono_thr
    mean_diff = np.mean(thr_diff[:])
    min_diff = np.min(thr_diff[:])
    max_diff = np.max(thr_diff[:])
    print('Min/max/mean differences: ', min_diff, ' , ', max_diff, ' , ', mean_diff)

    # # set up 2D interpolation
    rp_curt = rpos_vals[0:-2]
    xnew = np.linspace(rpos_vals[1], rpos_vals[-1], 50)
    ynew = np.linspace(surv_vals[1], surv_vals[-1], 50)
    np.meshgrid(xnew, ynew)

    f_interp = interpolate.interp2d(rp_curt, surv_vals, mono_thr[:, 0:-2])
    znew_mp = f_interp(xnew, ynew)
    m_min = np.min(znew_mp)
    m_max = np.max(znew_mp)

    f_interp = interpolate.interp2d(rp_curt, surv_vals, tripol_thr[:, 0:-2])
    znew_tp = f_interp(xnew, ynew)
    t_min = np.min(znew_tp)
    t_max = np.max(znew_tp)

    all_min = np.min([t_min, m_min])
    all_max = np.max([t_max, m_max])

    # rounding manually
    all_min = 25.0
    all_max = 85.0
    n_levels = 6
    labels = ['P1', 'P2', 'P3', 'P4']

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3, w_pad=2, h_pad=2.0)
    if filled_contours:
        CS3 = axs[0, 0].contourf(1 - rpos_vals, surv_vals, mono_thr,
                                 np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                 cmap='viridis', extend='both')
        # CS3 = axs[0].contourf(rpos_vals, surv_vals, mono_thr, cmap='hot')
        low_rpos_val = -0.5
        high_rpos_val = 0.5
        low_surv_val = 0.4
        high_surv_val = 0.8
        axs[0, 0].set_xlabel('Electrode distance (mm)')
        axs[0, 0].set_ylabel('Fractional neuronal density')
        axs[0, 0].set_title('Monopolar', fontsize=12)
        lab_shift = 0.025
        axs[0, 0].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].plot([1 - high_rpos_val], [high_surv_val], 'sk', markersize=20)
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='blue')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='blue',
                       linestyle='dashed')
        axs[0, 0].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black',
                       linestyle='dashed')
        axs[0, 0].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black')
        axs[0, 0].set_xticks([0.4, 0.8, 1.2, 1.6])

        cs4 = axs[0, 1].contourf(1 - rpos_vals, surv_vals, tripol_thr,
                                 np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                 cmap='viridis', extend='both')
        axs[0, 1].set_title('Tripolar', fontsize=12)
        axs[0, 1].set_xlabel('Electrode distance (mm)')
        axs[0, 1].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='red')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='red',
                       linestyle='dashed')
        axs[0, 1].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray',
                       linestyle='dashed')
        axs[0, 1].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray')
        axs[0, 1].set_xticks([0.4, 0.8, 1.2, 1.6])
    else:
        CS3 = axs[0, 0].contour(1 - rpos_vals, surv_vals, mono_thr,
                                np.arange(all_min, all_max, (all_max - all_min) / n_levels), colors='k')

        # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
        # then adds a percent sign.
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} dB" if plt.rcParams["text.usetex"] else f"{s} dB"

        manual_locations = [(0.3, 0.6), (1.0, 0.6), (1.0, 0.2), (1.2, 0.05)]
        axs[0, 0].clabel(CS3, CS3.levels, inline=True, fmt=fmt, fontsize=10, manual=manual_locations)
        low_rpos_val = -0.5
        high_rpos_val = 0.5
        low_surv_val = 0.4
        high_surv_val = 0.8
        axs[0, 0].set_xlabel('Electrode distance (mm)')
        axs[0, 0].set_ylabel('Fractional neuronal density')
        axs[0, 0].set_title('Monopolar', fontsize=12)
        lab_shift = 0.025
        axs[0, 0].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - high_rpos_val - 0.15, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='blue')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='blue',
                       linestyle='dashed')
        axs[0, 0].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black',
                       linestyle='dashed')
        axs[0, 0].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black')
        axs[0, 0].set_xticks([0.4, 0.8, 1.2, 1.6])

        cs4 = axs[0, 1].contour(1 - rpos_vals, surv_vals, tripol_thr,
                                np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                extend='both', colors='k')
        manual_locations = [(0, 0.9), (0.2, 0.6), (0.2, 0.2), (0.7, 0.6), (0.9, 0.2), (1.3, 0.1)]
        axs[0, 1].clabel(cs4, cs4.levels, inline=True, fmt=fmt, fontsize=10, manual=manual_locations)
        axs[0, 1].set_title('Tripolar', fontsize=12)
        axs[0, 1].set_xlabel('Electrode distance (mm)')
        axs[0, 1].text(1 - high_rpos_val - 0.15, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='red')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='red',
                       linestyle='dashed')
        axs[0, 1].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray',
                       linestyle='dashed')
        axs[0, 1].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray')
        axs[0, 1].set_xticks([0.4, 0.8, 1.2, 1.6])

    #  calculate  indices from target survival and rpos values
    low_rpos_idx = np.argmin(np.abs(rpos_vals - low_rpos_val))
    high_rpos_idx = np.argmin(np.abs(rpos_vals - high_rpos_val))
    low_surv_idx = np.argmin(np.abs(surv_vals - low_surv_val))
    high_surv_idx = np.argmin(np.abs(surv_vals - high_surv_val))

    axs[1, 0].plot(1 - rpos_vals, mono_thr[high_surv_idx, :], color='blue', linestyle='solid')
    axs[1, 0].plot(1 - rpos_vals, tripol_thr[high_surv_idx, :], color='red', linestyle='solid')
    axs[1, 0].plot(1 - rpos_vals, mono_thr[low_surv_idx, :], color='blue', linestyle='dashed')
    axs[1, 0].plot(1 - rpos_vals, tripol_thr[low_surv_idx, :], color='red', linestyle='dashed')
    axs[1, 0].axes.set_xlabel('Electrode distance (mm)')
    axs[1, 0].axes.set_ylabel('Threshold (dB)')
    axs[1, 0].axes.set_xlim([0.1, 1.9])
    axs[1, 0].axes.set_ylim([20, 80])
    axs[1, 0].set_xticks([0.4, 0.8, 1.2, 1.6])

    axs[1, 1].plot(surv_vals, mono_thr[:, high_rpos_idx], color='black', linestyle='solid')
    axs[1, 1].plot(surv_vals, tripol_thr[:, high_rpos_idx], color='gray', linestyle='solid')
    axs[1, 1].plot(surv_vals, mono_thr[:, low_rpos_idx], color='black', linestyle='dashed')
    axs[1, 1].plot(surv_vals, tripol_thr[:, low_rpos_idx], color='gray', linestyle='dashed')
    axs[1, 1].axes.set_xlabel('Fractional neuronal density')
    axs[1, 1].axes.set_xlim([0.1, 0.9])
    axs[1, 1].axes.set_ylim([20, 80])

    plt.savefig('Fig_2D_contour.eps', format='eps')

    figrows = 2
    figcols = 2
    fig2, axs2 = plt.subplots(figrows, figcols)
    idx_surv = 0
    idx_rpos = 0
    the_ax = 0
    for i in range(4):
        if i == 0:
            idx_surv = high_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[0, 0]
        elif i == 1:
            idx_surv = high_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[0, 1]
        elif i == 2:
            idx_surv = low_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[1, 0]
        elif i == 3:
            idx_surv = low_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[1, 1]

        this_mp_thr = [mono_thr[idx_surv, idx_rpos]]
        cont_mp = the_ax.contour(1 - rpos_vals, surv_vals, mono_thr, this_mp_thr, colors='blue')
        if i == 2 or i == 3:
            the_ax.axes.set_xlabel('Electrode distance (mm)', fontsize=14)

        if i == 0:
            the_ax.axes.set_ylabel('Fractional neuronal density', fontsize=14)
            the_ax.yaxis.set_label_coords(-0.2, -0.08)

        this_tp_thr = [tripol_thr[idx_surv, idx_rpos]]
        cont_tp = the_ax.contour(1 - rpos_vals, surv_vals, tripol_thr, this_tp_thr, colors='red')
        mpcontour = cont_mp.allsegs[0]
        tpcontour = cont_tp.allsegs[0]
        nmp = len(mpcontour[0])
        ntp = len(tpcontour[0])
        mpx = np.zeros(nmp)
        mpy = np.zeros(nmp)
        tpx = np.zeros(ntp)
        tpy = np.zeros(ntp)

        for j in range(0, nmp):  # Should be able to do this without for loops
            mpx[j] = mpcontour[0][j][0]
            mpy[j] = mpcontour[0][j][1]

        for j in range(0, ntp):
            tpx[j] = tpcontour[0][j][0]
            tpy[j] = tpcontour[0][j][1]

        x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)
        the_ax.plot(x[-1], y[-1], 'x', color='black', markersize='10', mew=2.5)
        if i > 0:
            the_ax.plot(x[0], y[0], 'x', color='gray', markersize='8')
        the_ax.set_xlim([0, 1.9])
        the_ax.text(0.1, 0.8, labels[i], fontsize=16)

    plt.savefig('Fig5_contour_examples.pdf', format='pdf')

    if map_unique_solutions:
        n_sols = np.zeros((nrpos, nsurv), dtype=int)
        # Map whether there is only a single solutions giving these values (or 0 or > 1)
        for sidx, surv in enumerate(surv_vals):
            # print('sidx = ', sidx)
            print('Approximately ', 100 * (sidx * nrpos) / (nrpos * nsurv), ' % done.')
            for ridx, rpos in enumerate(rpos_vals):
                # print('ridx is ', ridx)

                fig, ax1 = plt.subplots()
                ax1 = plt.contour(rpos_vals, surv_vals, mono_thr, [mono_thr[sidx, ridx]], colors='green')
                ax2 = plt.contour(rpos_vals, surv_vals, tripol_thr, [tripol_thr[sidx, ridx]], colors='red')
                mpcontour = ax1.allsegs[0]
                tpcontour = ax2.allsegs[0]
                # if ifPlotContours == False:
                #    plt.close(fig)

                nmp = len(mpcontour[0])
                ntp = len(tpcontour[0])
                mpx = np.zeros(nmp)
                mpy = np.zeros(nmp)
                tpx = np.zeros(ntp)
                tpy = np.zeros(ntp)

                for j in range(0, nmp):  # Should be able to do this without for loops
                    mpx[j] = mpcontour[0][j][0]
                    mpy[j] = mpcontour[0][j][1]

                for j in range(0, ntp):
                    tpx[j] = tpcontour[0][j][0]
                    tpy[j] = tpcontour[0][j][1]

                # try finding intersection using Shapely class
                if len(mpcontour[0]) == 1 or len(tpcontour[0]) == 1:
                    n_sols[ridx, sidx] = 0
                else:
                    ('len contours: ', len(mpcontour[0]), len(tpcontour[0]))
                    line1 = shap.LineString(mpcontour[0])
                    line2 = shap.LineString(tpcontour[0])
                    x = shap.intersection(line1, line2)

                    # How many intersections? 0, 1 or more?  If single intersection use those values
                    if x.geom_type == 'Point':  # test for multiple identical values

                        n_sols[ridx, sidx] = 1

                    elif x.geom_type == 'MultiPoint':
                        print('Multipoint at sidx = ', sidx, ' and ridx: ', ridx, ' : ', x)
                        n_sols[ridx, sidx] = 2
                    else:
                        n_sols[ridx, sidx] = 0

        solmap_file = FWDOUTPUTDIR + 'solution_map' + STD_TEXT
        np.savez(solmap_file, surv_vals, rpos_vals, n_sols)
        fig_nsols, ax_nsols = plt.subplots()
        cs_nsols = ax_nsols.contour(rpos_vals, surv_vals, n_sols, [1, 2])

    plt.show()


if __name__ == '__main__':
    fig_2D_contour()
