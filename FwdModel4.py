# Forward Model version 4. 15 April 2022

import pickle
import datetime
import os
import csv
import matplotlib.pyplot as plt
#  These are local to this project
import set_scenario as s_scen
import survFull
import getThresholds as gT
from common_params import *


# Needs cleanup
FASTZ = True  # use an interpolation method along z-axis

# We depend on a voltage and activation tables calculated using
# voltage_calc.py and saved as a .dat file. the file is specified in commonParams
with open(FIELDTABLE, 'rb') as combined_data:
    data = pickle.load(combined_data)
    combined_data.close()

fp = data[0]
fp['zEval'] = np.array(fp['zEval'])
vVals = data[1]
act_vals = data[2]
GRID['table'] = act_vals

COCHLEA['res1'] = fp['resInt'] * np.ones(NELEC)  # Note these values do not match those of Goldwyn et al., 2010
COCHLEA['res2'] = fp['resExt'] * np.ones(NELEC)  # resistivities are in Ohm*cm (conversion to Ohm*mm occurs later)
GRID['r'] = fp['rspace']  # only 1 of the 3 cylindrical dimensions can be a vector (for CYLINDER3D_MAKEPROFILE)

ifPlot = True  # Whether to plot the results
sigmaVals = [0, .9]
nSig = len(sigmaVals)

# Automatically create scenarios with uniform conditions across electrode positions
# We've mostly abandoned this for customized scenarios from set_scenario()
# survScenarios = [ 75]
# rposScenarios = [-0.5, 0.0, 0.5]

# nSurvS = len(survScenarios)
# nRposS = len(rposScenarios)

# nScenarios = nSurvS*nRposS
# scenarios = []

# for i in range(0, nSurvS):
#     for j in range(0, nRposS):
#         tempval = str(rposScenarios[j])
#         print('tempval = ', tempval)
#         scenarios.append('Uniform'+str(survScenarios[i])+'R'+tempval.replace('.', ''))

# print(scenarios)

for scenario in scenarios:
    [survVals, ELECTRODES['rpos']] = s_scen.set_scenario(scenario, NELEC)

    if not os.path.isdir(FWDOUTPUTDIR):
        os.makedirs(FWDOUTPUTDIR)

    OUTFILE = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'

    # Additional setup
    RUN_INFO['scenario'] = scenario
    RUN_INFO['run_time'] = datetime.datetime.now()
    COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # note that '.rpos' is in mm, must fit inside the radius

    # Construct the simParams structure
    simParams['cochlea'] = COCHLEA
    simParams['electrodes'] = ELECTRODES
    simParams['channel'] = CHANNEL
    simParams['grid'] = GRID
    simParams['run_info'] = RUN_INFO

    nZ = len(GRID['z'])  # Convenience variable

    # Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc,
    # made above % Keep in mind that the activation sensitivity will be FIXED, as will the number of neurons required
    # for threshold. Therefore, the final current to achieve theshold will vary according to the simple minimization
    # routine. Also note this works much faster when the field calculations are performed with a look-up table.
    avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
    rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
    rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
    rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling

    # Specify which variables to vary and set up those arrays
    thr_sim_db = np.empty((NELEC, nSig))  # Array for threshold data for different stim elecs and diff sigma values
    thr_sim_db[:] = np.nan

    # Get survival values for all 330 clusters from the 16 values at electrode
    # positions.
    NEURONS['nsurvival'] = survFull.survFull(simParams['electrodes']['zpos'], survVals, simParams['grid']['z'])
    NEURONS['rlvl'] = rlvltable
    simParams['neurons'] = NEURONS
    # Sanity check. Could add other sanity checks here
    # if any(simParams.grid.r < 1):
    # raise('Ending script. One or more evaluation points are inside cylinder; not appropriate for neural activation.')

    # Determine threshold for each value of sigma
    for i in range(0, nSig):  # number of sigma values to test
        simParams['channel']['sigma'] = sigmaVals[i]
        [thr_sim_db[:, i], neuron_vals] = gT.getThresholds(act_vals, fp, simParams)

    # Write a csv file
    with open(OUTFILE, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        for row in range(0, NELEC):
            data_writer.writerow([row, survVals[row], ELECTRODES['rpos'][row], thr_sim_db[row, 0], thr_sim_db[row, 1]])
    data_file.close()

    # Save simParams
    # np.save(FWDOUTPUTDIR + 'simParams' + scenario, simParams, allow_pickle=True)
    spname = FWDOUTPUTDIR + 'simParams' + scenario
    with open(spname + '.pickle', 'wb') as f:
        pickle.dump(simParams, f, pickle.HIGHEST_PROTOCOL)
    # Note that this is saving only the last simParams structure from the loops on sigma and in getThresholds.

    # Plot the results, if desired
    if ifPlot:
        fig1, ax1 = plt.subplots()
        ax1.plot(np.arange(0, NELEC) + 1, thr_sim_db, marker='o')
        titleText = 'Threshold ' + scenario
        ax1.set(xlabel='Electrode number', ylabel='Threshold (dB)', title=titleText)

        plt.show()
# Save PDF, if desired
#        legend([simSurv, targetSurv], 'sim', 'target', 'Location', 'north');
#        print('-dpdf', '-painters', '-bestfit', 'epsFig.pdf');
#        movefile('epsFig.pdf', [FWDOUTPUTDIR scenario '_thresh.pdf']);
