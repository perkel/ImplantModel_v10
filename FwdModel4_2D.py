# Forward model 2D version

# This 2D version uses a single electrode and scans through an arbitrary set of values for survival values and
# electrode position
# It is critical to note that the parameter values do not to be tied to the # of electrodes in an implant
# David J. Perkel 7 March 2019
# Translated to python 8 March 2020
# Reorganizing the code to be more modular and to allow sharing of code
# with inverse model, for simplicity and reproducibility
# This version runs through a set of 'scenarios', each with a specific set
# of electrode radial positions and neuron survival values. For each, it
# calculates predicted threshold for monopolar and partial tripolar
# stimulation, typically sigma = 0 and 0.9.
# Re-written 12 August 2021 to simplify data structures into dicts.

import survFull
import getThresholds as gT
import datetime
import os
import csv
import matplotlib.pyplot as plt
import pickle
from common_params import *
import fig_2D_contour
import PlotNeuronActivation

# We depend on a voltage and activation tables calculated using
# voltage_calc.py and saved as a .dat file
with open(FIELDTABLE, 'rb') as combined_data:
    data = pickle.load(combined_data)
    combined_data.close()

# Unpack data from the field data file
fp = data[0]
vVals = data[1]
actVals = data[2]
# Convert zEval to a numpy array
fp['zEval'] = np.array(fp['zEval'])

GRID['table'] = actVals

COCHLEA['res1'] = fp['resInt'] * np.ones(NELEC)  # Note these valus do not match thos of Goldwyn et al., 2010
COCHLEA['res2'] = fp['resExt'] * np.ones(NELEC)  # resistivities are in Ohm*cm (conversion to Ohm*mm occurs later)
GRID['r'] = fp['rspace']  # only 1 of the 3 cylindrical dimensions can be a vector (for CYLINDER3D_MAKEPROFILE)
COEF = 0  # convex: -1 | 0.4, 0.9 ; linear: 0 | 1; concave: +1 | 1.0, 1.8
POW = 1

SIDELOBE = 1
################################

ifPlot = True  # Whether to plot the results
ifPlotContours = False
survVals = np.arange(0.04, 0.97, 0.02)  # Was 0.02
rposVals = np.arange(-0.95, 0.96, 0.02)  # Was 0.02
hires = '_hi_res'
nSurv = len(survVals)
nRpos = len(rposVals)

# set up filename
descrip = "surv_" + str(np.min(survVals)) + "_" + str(np.max(survVals)) + "_rpos_" +\
          str(np.min(rposVals)) + "_" + str(np.max(rposVals)) + hires

if not os.path.isdir(FWDOUTPUTDIR):
    os.mkdir(FWDOUTPUTDIR)

OUTFILE = FWDOUTPUTDIR + 'FwdModelOutput_' + descrip + '.csv'

# Additional setup
COCHLEA['timestamp'] = datetime.datetime.now()
ELECTRODES['timestamp'] = datetime.datetime.now()
CHANNEL['timestamp'] = datetime.datetime.now()
COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # rpos is in mm, so be sure they fit inside the radius

# Construct the simParams structure
simParams['cochlea'] = COCHLEA
simParams['electrodes'] = ELECTRODES
simParams['channel'] = CHANNEL
simParams['grid'] = GRID

# Convenience variables
nZ = len(GRID['z'])

# Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc,
# made above. Keep in mind that the activation sensitivity will be FIXED, as will the number of neurons required
# for threshold. Therefore, the final current to achieve theshold will vary according to the simple minimization
# routine. Also note this works much faster when the field calculations are performed with a look-up table.
avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling

# Specify which variables to vary and set up those arrays
sigmaVals = [0, .9]
nSig = len(sigmaVals)
thr_sim_db = np.empty((nSurv, nRpos, nSig))  # Array for threshold data for different stim elecs and diff sigma values
thr_sim_db[:] = np.nan
neuronact = np.empty((nSurv, nRpos, nSig, 1, len(GRID['z'])))  # Only 1 electrode in this model
neuronact[:] = np.nan
n_sols = np.zeros((nSurv, nRpos), dtype=int)

# Get survival values for all 330 clusters from the 16 values at electrode positions.
simParams['neurons']['rlvl'] = rlvltable

# Sanity check. Could add other sanity checks here
# if any(simParams.grid.r < 1):
# raise('Ending script. One or more evaluation points are inside cylinder; not appropriate for neural activation.')

# Determine threshold for each value of sigma, looping through survival values and radial positions
for j, surv in enumerate(survVals):
    print('surv  ', j, ' of ', len(survVals))
    # Get survival values for all 330 clusters from the 16 values at electrode positions
    simParams['neurons']['nsurvival'] = survFull.survFull(simParams['electrodes']['zpos'], surv, simParams['grid']['z'])

    for k, rpos in enumerate(rposVals):
        simParams['electrodes']['rpos'] = rpos

        for i in range(0, nSig):  # number of sigma values to test
            simParams['channel']['sigma'] = sigmaVals[i]
            tempvals = gT.getThresholds(actVals, fp, simParams)
            nexttemp = tempvals[0]
            nexttemp2 = tempvals[1]
            thr_sim_db[j, k, i] = nexttemp
            if np.max(nexttemp2) == 0:
                print("flat activation")
            neuronact[j, k, i, :, :] = nexttemp2
            neuroncount = np.sum(neuronact[j, k, i, :, :])

# Write results to a CSV file
header1 = ['Monopolar thresholds', 'rpos values in columns']
header2 = ['Survival ']
for i, rpos in enumerate(rposVals):
    header2 += str(rpos) + ', '

with open(OUTFILE, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
    data_writer.writerow(header1)
    data_writer.writerow(header2)
    for row, s_val in enumerate(survVals):
        data_writer.writerow([row, s_val, thr_sim_db[row, 0]])

    header1 = 'Tripolar thresholds: rpos values in columns; sigma = ' + str(sigmaVals[-1])
    data_writer.writerow(header1)
    data_writer.writerow(header2)
    for row, s_val in enumerate(survVals):
        data_writer.writerow([row, s_val, thr_sim_db[row, 1]])

data_file.close()

np.savetxt(FWDOUTPUTDIR + 'Monopolar_2D_' + STD_TEXT + '.csv', thr_sim_db[:, :, 0], delimiter=',')
np.savetxt(FWDOUTPUTDIR + 'Tripolar_09_2D_' + STD_TEXT + '.csv', thr_sim_db[:, :, 1], delimiter=',')

np.save(FWDOUTPUTDIR + 'simParams' + descrip, simParams)
# Note that this is saving only the last simParams structure from the loops on sigma and in getThresholds.

# display min and max threshold values
print('min MP thr:  ', np.min(thr_sim_db[:, :, 0]))
print('max MP thr:  ', np.max(thr_sim_db[:, :, 0]))
print('min TP thr:  ', np.min(thr_sim_db[:, :, 1]))
print('max TP thr:  ', np.max(thr_sim_db[:, :, 1]))

# Save neuron activation data into a binary file
temp0 = [(survVals, rposVals)]
temp1 = [temp0, neuronact]
np.savez(FWDOUTPUTDIR + 'neuronact_' + STD_TEXT, survVals, rposVals, neuronact)

# Plot the results
if ifPlot:
    # fig, ax = plt.subplots()
    # ax.plot(rposVals, thr_sim_db[2, :, 0], marker='o')
    # titleText = 'Threshold ' + descrip
    # ax.set(xlabel='Electrode position (mm)', ylabel='Threshold (dB)', title=titleText)
    #
    # fig2, ax2 = plt.subplots()
    # ax2.plot(survVals, thr_sim_db[:, 10, 0], marker='o', color='r')
    # ax2.plot(survVals, thr_sim_db[:, 20, 0], marker='o', color='b')
    # ax2.plot(survVals, thr_sim_db[:, 30, 0], marker='o', color='g')
    # titleText = 'Threshold ' + descrip
    # ax2.set(xlabel='Survival fraction', ylabel='Threshold monopolar (dB)', title=titleText)
    #
    # fig3, ax3 = plt.subplots()
    # ax3.plot(survVals, thr_sim_db[:, 10, 1], marker='o', color='r')
    # ax3.plot(survVals, thr_sim_db[:, 20, 1], marker='o', color='b')
    # ax3.plot(survVals, thr_sim_db[:, 30, 1], marker='o', color='g')
    # titleText = 'Threshold ' + descrip
    # ax3.set(xlabel='Survival fraction', ylabel='Threshold tripolar (dB)', title=titleText)

    fig_2D_contour.fig_2D_contour()
    PlotNeuronActivation.PlotNeuronActivation()
    plt.show()
