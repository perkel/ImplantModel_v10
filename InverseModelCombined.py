# InverseModelCombined.py
# Script to fit TP and MP threshold data to the Goldwyn, Bierer, and
# Bierer cochlear activation model.

# Original code by Steve Bierer.

# Translated to python 3 and adapted to fit both survival and rpos values by David J. Perkel December 2020

# Modified 6 August 2021 by DJP to start by fitting a single electrode at a time. This gives initial conditions
# for each electrode. Then there's a holistic fitting process using all electrode parameters

import cProfile
import io
import pstats
from pstats import SortKey
import csv
import os
import scipy.signal as sig
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from lmfit import Minimizer, Parameters, report_fit
import intersection as intsec

# local files
import set_scenario as s_scen
import load_fwd_csv_data as lcsv
import survFull
import getThresholds as gT
import subject_data
import PlotInverseResults
from common_params import *  # import common values across all models

# User adjustable parameters
fit_mode = 'combined'  # Which variable(s) to fit? Alternatives are 'combined', 'rpos' or 'survival'
ifPlot = True  # Whether to plot output at end
unsupervised = True  # Makes and saves summary plots but does not display them and wait for user input before proceeding
ifPlotGuessContours = False  # Option to plot initial guesses for parameters given to the fitting algorithm
use_fwd_model = True  # If True, use output from the forward model. If False, use subject data
fit_tol = 0.01  # Fit tolearnce for subject fits


# For optimizing fit to thresholds need e_field, sim_params, sigvals
# This is for a single electrode
def objectivefunc_lmfit(p, sigvals, sim_params, f_par, e_field, thr_goals, this_elec):
    nel = len(sim_params['electrodes']['zpos'])
    vals = p.valuesdict()
    show_retval = False

    sim_params['electrodes']['rpos'] = vals['rpos_val']
    tempsurv = np.zeros(nel)
    tempsurv[:] = vals['surv_val']
    print('tempsurv = ', tempsurv)

    sim_params['neurons']['nsurvival'] = survFull.survFull(sim_params['electrodes']['zpos'],
                                                           tempsurv, simParams['grid']['z'])

    # Call for monopolar then tripolar
    sim_params['channel']['sigma'] = sigvals[0]
    thresh_mp = gT.getThresholds(e_field, f_par, sim_params)
    sim_params['channel']['sigma'] = sigvals[1]
    thresh_tp = gT.getThresholds(e_field, f_par, sim_params)
    mp_err = np.abs(thresh_mp[0] - thr_goals['thrmp_db'][this_elec])
    tp_err = np.abs(thresh_tp[0] - thr_goals['thrtp_db'][this_elec])
    if np.isnan(tp_err):
        tp_err = 0.0
    mean_error = (mp_err + tp_err) / 2.0
    if show_retval:
        print('Mean error (dB) = ', mean_error)

    retval = [mp_err, tp_err]
    return retval


# For optimizing fit to thresholds need e_field, sim_params, sigvals
# This is for all electrodes at once
def objectivefunc_lmfit_all(par, sigvals, sim_params, f_par, e_field, thr_goals):
    # Repack parameters into arrays
    nel = len(sim_params['electrodes']['zpos'])
    vals = par.valuesdict()
    show_retval = True  # helpful to track progress of the fitting process

    sim_params['electrodes']['rpos'] = np.zeros(nel)
    for i in range(0, nel):
        varname = 'v_%i' % i
        myvalue = vals[varname]
        sim_params['electrodes']['rpos'][i] = myvalue

    tempsurv = np.zeros(nel)
    for i, loopval in enumerate(range(nel, 2 * nel)):
        varname = 'v_%i' % (i + nel)
        myvalue = vals[varname]
        tempsurv[i] = myvalue

    sim_params['neurons']['nsurvival'] = survFull.survFull(sim_params['electrodes']['zpos'],
                                                           tempsurv, simParams['grid']['z'])

    # Call for monopolar then tripolar
    sim_params['channel']['sigma'] = sigvals[0]
    thresh_mp = gT.getThresholds(e_field, f_par, sim_params)
    sim_params['channel']['sigma'] = sigvals[1]
    thresh_tp = gT.getThresholds(e_field, f_par, sim_params)
    # Calculate errors
    mp_err = np.nanmean(np.abs(np.subtract(thresh_mp[0], thr_goals['thrmp_db'])))
    tp_err = np.nanmean(np.abs(np.subtract(thresh_tp[0][1:nel - 1], thr_goals['thrtp_db'][1:nel - 1])))
    mean_error = (mp_err + tp_err) / 2.0
    # retval = np.append(np.subtract(thresh_mp[0], thr_goals['thrmp_db']), 0)
    mp_diff = np.subtract(thresh_mp[0], thr_goals['thrmp_db'])
    tp_diff = np.subtract(thresh_tp[0][1:nel - 1], thr_goals['thrtp_db'][1:nel - 1])
    tempzero = np.zeros(1)
    retval = np.concatenate((mp_diff, tempzero, tp_diff, tempzero))
    # Returns a vector of errors, with the first and last of the tripolar errors set to zero
    # because they can't be calculated
    if show_retval:  # helpful for debugging
        scen = simParams['run_info']['scenario']
        print('subj/scen: ', scen, '; tempsurv[4] = ', '%.3f' % tempsurv[4], '; rpos[4]= ',
              '%.3f' % sim_params['electrodes']['rpos'][4],
              '; Mean error (dB) = ', '%.3f' % mean_error, '; Max error (dB) = ', '%.3f' % np.nanmax(np.abs(retval)))
    return retval


def max_diff_adjacent(x, pars):
    # pars 0-15 are for distance
    diff = np.diff(pars[0:16])
    abs_diff = np.abs(diff)
    max_abs_diff = np.max(abs_diff)
    return max_abs_diff


def find_closest(x1, y1, x2, y2):  # returns indices of the point on each curve that are closest
    # Brute force (pretty ugly, but hopefully a rare case)
    # TODO Should test for case where x1 or x2 is a scalar
    n1 = len(x1)
    n2 = len(x2)

    min_dist = 5.0
    min_idx = [0, 0]
    for ii in range(n1):
        for jj in range(n2):
            dist = np.sqrt((((x1[ii] - x2[jj]) / 2) ** 2) + ((y1[ii] - y2[jj]) ** 2))
            if dist < min_dist:
                min_idx = [ii, jj]
                min_dist = dist

    return min_idx


def inverse_model_combined_se():  # Start this script
    if not os.path.isdir(INV_OUT_PRFIX):
        os.mkdir(INV_OUT_PRFIX)
        os.mkdir(INVOUTPUTDIR)
    else:
        if not os.path.isdir(INVOUTPUTDIR):
            os.path.mkdir(INVOUTPUTDIR)

    # First make sure that the 2D forward model has been run and load the data
    # It would be ideal to double check that it's the correct 2D data with identical parameters
    # Load monopolar data
    pr = cProfile.Profile()
    pr.enable()  # Start the profiler
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

    # Now hold these data until about to fit a particular combination of monopolar and tripolar threshold values
    # to see if there is more than one solution
    surv_grid_vals = np.arange(0.04, 0.97, 0.02)
    rpos_grid_vals = np.arange(-0.95, 0.96, 0.02)

    # Open field data and load data
    if "fieldTable" not in locals():
        with open(FIELDTABLE, 'rb') as combined_data:
            data = pickle.load(combined_data)
            combined_data.close()

            #  load(FIELDTABLE) # get model voltage/activating tables, if not already loaded
            # (output is 'fieldTable' and 'fieldParams')
            fp = data[0]
            # Temp fixup
            fp['zEval'] = np.array(fp['zEval'])
            act_vals = data[2]  # the data[1] has voltage values, which we are ignoring here
            simParams['grid']['table'] = act_vals

    num_scen = len(scenarios)
    # Set up array for summary values
    thresh_err_summary = np.zeros((num_scen, 2))
    rpos_summary = []
    rpos_err_summary = np.zeros(num_scen)
    if not use_fwd_model:
        dist_corr = np.zeros(num_scen)
        dist_corr_p = np.zeros(num_scen)

    for scen in range(0, len(scenarios)):
        scenario = scenarios[scen]
        simParams['run_info']['scenario'] = scenario
        if use_fwd_model:
            [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'
            [thr_data, ct_data] = lcsv.load_fwd_csv_data(csv_file)
        else:  # use threshold data from a subject
            SUBJECT = scenario
            ELECTRODES['rpos'] = np.zeros(NELEC)
            rposvals = ELECTRODES['rpos']
            thr_data = {'thrmp_db': (subject_data.subj_thr_data(SUBJECT))[0], 'thrmp': [],
                        'thrtp_db': (subject_data.subj_thr_data(SUBJECT))[1], 'thrtp': [], 'thrtp_sigma': 0.9}
            thr_data['thrtp_db'] = np.insert(thr_data['thrtp_db'], 0, np.NaN)  # put NaNs at ends of array
            thr_data['thrtp_db'] = np.append(thr_data['thrtp_db'], np.NaN)

            # Calculate offset to get closer to what the model can produce
            mp_offset_db = np.nanmean(thr_data['thrmp_db']) - np.nanmean(mono_thr)
            tp_offset_db = np.nanmean(thr_data['thrtp_db']) - np.nanmean(tripol_thr)
            # Use the monopolar offset
            overall_offset_db = mp_offset_db
            # overall_offset_db = 0
            # Use the mean of monopolar and tripolar offsets
            # overall_offset_db = np.mean([mp_offset_db, tp_offset_db])
            offset_mult = 1.0
            thr_data['thrmp_db'] -= offset_mult * overall_offset_db
            thr_data['thrtp_db'] -= offset_mult * overall_offset_db

            survvals = np.empty(NELEC)
            survvals[:] = np.nan
            ct_data = {'stdiameter': [], 'scala': [], 'elecdist': [], 'espace': 1.1, 'type': [], 'insrt_base': [],
                       'insert_apex': []}
            radius = 1.0
            ct_data['stdiameter'] = radius * 2.0 * (np.zeros(NELEC) + 1.0)

        #  rposvals = ELECTRODES['rpos']  # save this for later
        saverposvals = rposvals

        cochlea_radius = ct_data['stdiameter'] / 2.0
        if np.isnan(thr_data['thrtp_sigma']) or thr_data['thrtp_sigma'] < 0.75 or thr_data['thrtp_sigma'] > 1:
            print('The sigma value for the TP configuration is invalid.')
            exit()

        fradvec = (ct_data['stdiameter'] / 2)  # smooth the radius data!!
        fradvec = sig.filtfilt(np.hanning(5) / sum(np.hanning(5)), 1, fradvec)
        simParams['cochlea']['radius'] = fradvec
        avec = np.arange(0, 1.005, 0.01)  # create the neuron count to neuron spikes transformation
        rlvec = NEURONS['coef'] * (np.power(avec, 2.0)) + (1 - NEURONS['coef']) * avec
        rlvec = NEURONS['neur_per_clust'] * np.power(rlvec, NEURONS['power'])
        rlvltable = np.stack((avec, rlvec))
        simParams['neurons']['rlvl'] = rlvltable

        # Construct the simParams structure
        simParams['cochlea'] = COCHLEA
        simParams['electrodes'] = ELECTRODES
        simParams['channel'] = CHANNEL
        simParams['channel']['sigma'] = 0.0
        thresholds = np.empty(NELEC)  # Array to hold threshold data for different simulation values
        thresholds[:] = np.nan
        fitrposvals = np.zeros(NELEC)
        fitsurvvals = np.zeros(NELEC)
        par = Parameters()
        par._asteval.symtable['max_diff_adjacent'] = max_diff_adjacent
        initvec = []

        if fit_mode == 'combined':  # Optimize survival and rpos to match MP and TP thresholds
            # Loop on electrodes, fitting rpos and survival fraction at each location
            nsols = np.zeros(NELEC)  # number of solutions found from the 2D maps

            for i in range(1, NELEC - 1):  # Fit params for each electrode (except ends, where there is no tripol value)
                mptarg = thr_data['thrmp_db'][i]
                tptarg = thr_data['thrtp_db'][i]

                # Get contours and find intersection to find initial guess for overall fitting
                # fig3, ax3 = plt.subplots()
                # rp_curt = rpos_grid_vals[0:-2]  # curtailed rpos values
                # f_interp_mp = interpolate.interp2d(rp_curt, surv_grid_vals, mono_thr[:, 0:-2])
                # f_interp_tp = interpolate.interp2d(rp_curt, surv_grid_vals, tripol_thr[:, 0:-2])
                # xnew = np.linspace(rpos_grid_vals[0], rpos_grid_vals[-2], 50)
                # ynew = np.linspace(surv_grid_vals[0], surv_grid_vals[-2], 50)
                # xn, yn = np.meshgrid(xnew, ynew)
                # znew_mp = f_interp_mp(xnew, ynew)
                # znew_tp = f_interp_tp(xnew, ynew)
                # ax3 = plt.contour(xn, yn, znew_mp, [mptarg], colors='green')
                # ax3.axes.set_xlabel('Rpos (mm)')
                # ax3.axes.set_ylabel('Survival fraction')
                # ax4 = plt.contour(xn, yn, znew_tp, [tptarg], colors='red')
                # ax4.axes.set_xlabel('Rpos (mm)')
                # ax4.axes.set_ylabel('Survival fraction')
                # mpcontour = ax3.allsegs[0]  # Contour points in rpos x survival space that give this threshold
                # tpcontour = ax4.allsegs[0]

                fig4, ax5 = plt.subplots()
                ax5 = plt.contour(rpos_grid_vals, surv_grid_vals[2:], mono_thr[2:, :], [mptarg], colors='green')
                ax5.axes.set_xlabel('Rpos (mm)')
                ax5.axes.set_ylabel('Survival fraction')
                ax6 = plt.contour(rpos_grid_vals, surv_grid_vals[2:], tripol_thr[2:, :], [tptarg], colors='red')
                ax6.axes.set_xlabel('Rpos (mm)')
                ax6.axes.set_ylabel('Survival fraction')
                ax5.axes.xaxis.set_label('Rpos (mm)')
                ax5.axes.yaxis.set_label('Survival fraction')
                mpcontour = ax5.allsegs[0]
                tpcontour = ax6.allsegs[0]
                if ifPlotGuessContours == False:
                    plt.close(fig4)

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

                # How many intersections? 0, 1 or more?  If single intersection use those values
                nsols[i] = len(x)
                if nsols[i] == 0:
                    # no solution. This shouldn't happen with known scenarios,
                    # since the forward model calculated threshold, but it does happen in some cases.
                    if use_fwd_model:
                        print('no solution, but this is a known scenario')
                        exit()

                    # Maybe there's some error in this process.
                    # plt.show()

                    mp_idx, tp_idx = find_closest(mpx, mpy, tpx, tpy)

                    # rp_guess = mpx[mp_idx]
                    # sv_guess = mpy[mp_idx]
                    # rp_guess = tpx[tp_idx]  # Use tripolar best guess
                    # sv_guess = tpy[tp_idx]
                    rp_guess = (mpx[mp_idx] + tpx[tp_idx]) / 2.0
                    sv_guess = (mpy[mp_idx] + tpy[tp_idx]) / 2.0
                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')
                    print("no solutions. Closest: ", mp_idx, ' and ', tp_idx,
                          ' , leading to guesses of (position, survival): ', rp_guess, sv_guess)

                    # Temporary replacement
                    # rp_guess = 0.0
                    # sv_guess = 0.5
                    # print('No solutions. Guessing in middle of range')

                elif nsols[i] == 1:  # unique solution
                    print("one solution: ", x, y)
                    rp_guess = x
                    sv_guess = y
                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')

                else:  # multiple solutions
                    print(nsols[i], " solutions: ", x, ' and: ', y)
                    which_sols = np.zeros((4, int(nsols[i])))  # array for solutions and best fit
                    for sol in range(int(nsols[i])):  # Try all potential solutions; keep best
                        rp_guess = x[sol]
                        sv_guess = y[sol]

                        # Put variables into Parameters
                        # par.add('rpos_val', value=rp_guess, min=-0.8, max=0.8)
                        # par.add('surv_val', value=sv_guess, min=0.2, max=0.9)

                        # do fit, here with the default leastsq algorithm
                        # minner = Minimizer(objectivefunc_lmfit, par, fcn_args=(sigmaVals, simParams,
                        # fp, act_vals, thr_data, i))
                        # result = minner.minimize()
                        # which_sols[0, sol] = result.params['rpos_val']
                        # which_sols[1, sol] = result.params['surv_val']
                        # which_sols[2, sol] = np.mean(result.residual)
                        rposweight = 2
                        survweight = 0.0
                        if i > 1:  # calculate distance from previous coords
                            which_sols[3, sol] = np.sqrt(rposweight * (rp_guess - fitrposvals[i - 1]) ** 2 +
                                                         survweight * (sv_guess - fitsurvvals[i - 1]) ** 2)

                    # figure out which is best
                    print("which is best? use average")
                    # First attempt: pick rpos and surv that are closest to previous electrode (if there is one)
                    # min_val = np.amin(which_sols[3, :])
                    # closest_idx = np.where(which_sols[3, :] == min_val)[0]
                    # if len(closest_idx) > 1:
                    #     closest_idx = 0
                    # rp_guess = x[closest_idx]
                    # sv_guess = y[closest_idx]
                    # print('Closest is # ', closest_idx, ' , and guesses are: ', rp_guess, sv_guess)
                    # # print('And residual is: ', which_sols[2, closest_idx])
                    rp_guess = np.mean(x)
                    sv_guess = np.mean(y)

                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')

                fitrposvals[i] = rp_guess
                fitsurvvals[i] = sv_guess

            # fix values for first and last electrodes as identical to their neighbors
            fitrposvals[0] = fitrposvals[1]
            fitrposvals[-1] = fitrposvals[-2]
            fitsurvvals[0] = fitsurvvals[1]
            fitsurvvals[-1] = fitsurvvals[-2]

            initvec = np.append(fitrposvals, fitsurvvals)

            for i, val in enumerate(initvec):  # place values in to the par object
                if i < 16:
                    lb = -0.85  # lower and upper bounds
                    ub = 0.85
                else:
                    lb = 0.1
                    ub = 0.9

                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)

            start_pos = fitsurvvals  # prior to fitting

            # end block for single electrode fits during the combined fit

        elif fit_mode == 'rpos':  # fit rpos only; hold survival fixed as the values loaded from the scenario
            initvec = np.append(np.zeros(NELEC), survvals)
            for i, val in enumerate(initvec):
                if i < 16:
                    lb = -0.85  # lower and upper bounds
                    ub = 0.85
                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
                else:
                    par.add('v_%i' % i, value=initvec[i], vary=False)

            #  start_pos = initvec[NELEC:]
        elif fit_mode == 'survival':
            initvec = np.append(rposvals, (np.ones(NELEC) * 0.5))
            for i, val in enumerate(initvec):
                if i < 16:
                    lb = rposvals[i] - 0.01  # lower and upper bounds
                    ub = rposvals[i] + 0.01
                else:
                    lb = 0.1
                    ub = 0.9
                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)

            start_pos = initvec[NELEC:]

        # Contraint on maximum change of electrode posiiton for adjacen electrodes
        #         par.add('max_adj', expr=max_diff_adjacent(x, par), max=0.5)

        # Now do the main fitting for all electrodes at once
        minner = Minimizer(objectivefunc_lmfit_all, par, diff_step=0.02, nan_policy='omit',
                           fcn_args=(sigmaVals, simParams, fp, act_vals, thr_data))
        if use_fwd_model:
            result = minner.minimize(method='least_squares', diff_step=0.02)
        else:  # use CT data
            result = minner.minimize(method='least_squares', ftol=fit_tol, xtol=1e-3, max_nfev=2000)

        for i in range(NELEC):
            vname = 'v_%i' % i
            fitrposvals[i] = result.params[vname]
            vname = 'v_%i' % (i + NELEC)
            fitsurvvals[i] = result.params[vname]

        simParams['electrodes']['rpos'] = fitrposvals

        # report last fit
        report_fit(result)
        result.params.pretty_print()
        simParams['electrodes']['rpos'] = fitrposvals
        simParams['neurons']['nsurvival'] = survFull.survFull(simParams['electrodes']['zpos'], fitsurvvals,
                                                              simParams['grid']['z'])
        simParams['channel']['sigma'] = sigmaVals[0]
        thrsimmp = gT.getThresholds(act_vals, fp, simParams)
        simParams['channel']['sigma'] = sigmaVals[1]
        thrsimtp = gT.getThresholds(act_vals, fp, simParams)

        errvals = [np.subtract(thrsimmp[0], thr_data['thrmp_db']), np.subtract(thrsimtp[0][1:NELEC - 1],
                                                                               thr_data['thrtp_db'][1:NELEC - 1])]
        thrsim = [[thrsimmp[0]], [thrsimtp[0]]]
        thrtargs = [[thr_data['thrmp_db']], [thr_data['thrtp_db']]]

        # Summary data. Monopolar and tripolar errors saved separately.
        mean_mp_thr_err = np.nanmean(np.abs(np.array(thrsim[0]) - np.array(thrtargs[0])))
        mean_tp_thr_err = np.nanmean(np.abs(np.array(thrsim[1]) - np.array(thrtargs[1])))

        thresh_err_summary[scen, 0] = mean_mp_thr_err
        thresh_err_summary[scen, 1] = mean_tp_thr_err

        if use_fwd_model:
            [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)
            rposerrs = np.subtract(rposvals, fitrposvals)
            survivalerrs = np.subtract(survvals, fitsurvvals)
            rpos_summary.append([rposvals, fitrposvals])
            rpos_err_metric = np.NAN
            rpos_err_summary[scen] = np.NAN

            # Save values in CSV format
            save_file_name = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined.csv'
        else:
            ct_vals = subject_data.subj_ct_data(SUBJECT)
            survivalerrs = np.empty(NELEC)

            if np.any(ct_vals):
                rposerrs = np.subtract(fitrposvals, ct_vals)
                rpos_err_metric = np.mean(np.abs(rposerrs))
                rpos_summary.append([fitrposvals, ct_vals])
                rpos_err_summary[scen] = rpos_err_metric
                [dist_corr[scen], dist_corr_p[scen]] = stats.pearsonr(1 - ct_vals, 1 - fitrposvals)
            else:
                rposerrs = np.empty(NELEC)
                rpos_err_metric = np.NAN
                rpos_err_summary[scen] = np.NAN

            ##
            # Save values in CSV format
            save_file_name = INVOUTPUTDIR + SUBJECT + '_fitResults_' + 'combined.csv'
            # [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)

        # Save the data for this scenario
        with open(save_file_name, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
            if use_fwd_model:
                header = ['Electrode', 'Rposition', 'Survival', 'ThreshMP', 'ThreshTP',
                      'RpositionFit', 'SurvivalFit', 'RposError', 'SurvError', 'CT_position']
            else:
                header = ['Electrode', 'ThreshMP', 'ThreshTP', 'Fitted ThreshMP', 'Fitted ThreshTP',
                       'CT_position', 'RpositionFit', 'RposError', 'Fitted survival']

            data_writer.writerow(header)
            for row in range(0, NELEC):
                t1 = row
                t2 = rposvals[row]
                t3 = survvals[row]
                t4a = thrsim[0][0]
                t4 = t4a[row]
                t5a = thrsim[1][0]
                t5 = t5a[row]
                # t6 = opt_result.x[row]
                # t7 = opt_result.x[NELEC + row]
                t6 = fitrposvals[row]
                t7 = fitsurvvals[row]
                t8 = rposerrs[row]
                t9 = survivalerrs[row]
                if use_fwd_model:
                    data_writer.writerow([t1, t2, t3, t4, t5, t6, t7, t8, t9])
                else:
                    t10 = ct_vals[row]
                    data_writer.writerow([row, thrtargs[0][0][row], thrtargs[1][0][row], thrsim[0][0][row],
                                          thrsim[1][0][row], ct_vals[row], t6, t8, t7])
        data_file.close()

        # Save values in npy format
        save_file_name = INVOUTPUTDIR + scenario + '_fitResults_combined.npy'
        if use_fwd_model:
            np.save(save_file_name,
                    np.array([sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                     rposerrs, rpos_err_metric, survivalerrs], dtype=object))
        else:
            np.save(save_file_name,
                    np.array([sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                     rposerrs, rpos_err_metric, survivalerrs, ct_vals], dtype=object))

        np.save(INVOUTPUTDIR + scenario + '_simParams_inv', simParams)

        # Make plots
        if ifPlot:
            if use_fwd_model:
                txt_string = scenario
            else:
                txt_string = SUBJECT
            PlotInverseResults.plot_inverse_results(use_fwd_model, txt_string, unsupervised)

        # Save individual subject file
        # scenario, threshold error metric, and for CT cases: rpos error matric
        # print('saving this subject\'s fit data in CSV form')
        # if not use_fwd_model:
        #     summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_' + SUBJECT + '_data.csv'
        #     with open(summary_file_name, mode='w') as data_file:
        #         data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        #         header = ['Subject', 'Electrode', 'MP Threshold', 'Fitted MP Threshold', 'TP Threshold', 'Fitted TP Threshold',
        #                   'CT Position', 'Fitted Position',
        #                   'Dist correlation', 'Dist corr p']
        #         data_writer.writerow(header)
        #         for row in range(NELEC):
        #                 data_writer.writerow([scenario, row, '%.3f' % thresh_err_summary[row, 0],
        #                                       '%.3f' % thresh_err_summary[row, 1], '%.3f' % rpos_err_summary[row],
        #                                       '%.3f' % dist_corr[row], '%.5f' % dist_corr_p[row]])
        #         data_file.close()


    # Now we are done with the loop on scenarios/subjects
    pr.disable()  # stop the profiler
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(10)
    # print(s.getvalue())

    # Save the data into a single file
    print('Completed inverse model on ', num_scen, ' scenarios is total time (s) of: ', ps.total_tt)

    # save summary data in a CSV file
    # scenario, threshold error metric, and for CT cases: rpos error matric
    print('saving summary data in CSV form')
    summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.csv'
    with open(summary_file_name, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        header = ['Scenario/subject', 'Mean MP Threshold Error', 'Mean TP Threshold Error', 'Position Error',
                  'Dist correlation', 'Dist corr p']
        data_writer.writerow(header)
        for row in range(num_scen):
            if use_fwd_model:
                data_writer.writerow([scenarios[row], '%.3f' % thresh_err_summary[row, 0],
                                      '%.3f' % thresh_err_summary[row, 1], '%.3f' % rpos_err_summary[row]])
            else:
                data_writer.writerow([scenarios[row], '%.3f' % thresh_err_summary[row, 0],
                                      '%.3f' % thresh_err_summary[row, 1], '%.3f' % rpos_err_summary[row],
                                      '%.3f' % dist_corr[row], '%.5f' % dist_corr_p[row]])
        data_file.close()


    # save summary binary data file
    # scenario, threshold error metric, and for CT cases: rpos error matric
    summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.npy'
    np.save(summary_file_name, np.array([scenarios, rpos_summary], dtype=object))
    print('Done with saving summary files')


if __name__ == '__main__':
    inverse_model_combined_se()
