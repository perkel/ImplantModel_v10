#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:46:54 2020

@author: perkel
"""

# implant voltage calculations from Goldwyn et al, 2010, denovo.

# Import dependencies
import cProfile
import io
import pickle
import pstats
from datetime import datetime
from pstats import SortKey
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.special as spec
import remove_nans as r_nans


def goldwyn_beta(eps, k2, rs, re, n):
    krs = k2 * rs
    denom = ((eps * spec.kvp(n, krs) * spec.iv(n, krs)) - (spec.kn(n, krs) * spec.ivp(n, krs)))
    return spec.iv(n, k2 * re) / (denom * krs)


def goldwyn_phi(eps, k, rs, re, reval, n):  # Note Goldwyn alpha is zero for eval pt. outide cylinder
    phi = (goldwyn_beta(eps, k, rs, re, n) * spec.kn(n, k * reval))
    return phi


def integ_func(x, m_max, pratio, rad, reval, z, theta, relec):  # This is the Bessel function along z axis
    sum_contents = 0.0
    increments = np.zeros(m_max)
    rel_incrs = np.zeros(m_max)
    for idx in range(m_max):
        if idx == 0:
            gamma = 1.0
        else:
            gamma = 2.0

        increments[idx] = gamma * np.cos(idx * theta) * goldwyn_phi(pratio, x, rad, relec, reval, idx)
        sum_contents += increments[idx]
        rel_incrs[idx] = np.abs(increments[idx]) / sum_contents
        # print('m = ', idx, ' ; increment = ', increments[idx], ' ; rel_incr = ', rel_incrs[idx])

    return np.cos(x * z) * sum_contents


# See Rubinstein dissertation, 1988, Ch 6.

# Main parameters to vary
radius = 1.0  # cylinder radius
res_int = 70.0  # internal resistivity
res_ext = 250.0  # external resistivity

output_filename = '13June2023_MedResolution_Rext250.dat'

pr = cProfile.Profile()

# Field parameters. zEval can be higher (more precise, slower) or lower resolution (less precise, faster)
fp = {'model': 'cylinder_3d', 'fieldtype': 'Voltage_Activation', 'evaltype': 'SG', 'cylRadius': radius,
      'relec': np.arange(-0.95, 0.951, 0.05), 'resInt': res_int, 'resExt': res_ext, 'rspace': 1.3, 'theta': 0.0,
      # "High" resolution
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
      #              1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
      #              3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
      #              4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8,
      #              8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
      #              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
      #              22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0),
      # Medium resolution
      'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
                32.0, 34.0, 36.0, 38.0, 40.0),
      # Low resolution
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
      #           1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
      #           8.5, 9.0, 9.5, 10.0, 15, 20.0, 25.0, 30.0, 35.0, 40),
      # Very low resolution for debugging
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
      'mMax': 47, 'intStart': 1e-12, 'intEnd': 500.0, 'reval': 1.3,
      'ITOL': 1e-6, 'runDate': 'rundate', 'runOutFile': 'savefile', 'run_duration': 0.0}

now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
fp['runDate'] = date_time
fp['runOutFile'] = output_filename

# Derived values
resRatio = fp['resInt'] / fp['resExt']
rPrime = fp['reval']
nZ = len(fp['zEval'])  # this may be a problem if zEval is just a scalar
n_ypos = 3  # # of values to calculate across the y dimension for the 2nd spatial derivative

rElecRange = fp['relec']
nRElec = len(rElecRange)
voltageVals = np. zeros((nRElec, n_ypos, nZ))
activationVals = np.zeros((nRElec, nZ))
if_plot = False

pr.enable()  # Start the profiler

n_yeval = n_ypos // 2  # floor division
y_inc = 0.001  # 10 microns
yVals = np.arange(-n_yeval * y_inc, n_yeval * (y_inc * 1.01), y_inc)
transVoltage = np.empty(n_ypos)

# loop on electrode radial positions
for i, rElec in enumerate(rElecRange):
    # retval = integ_func(1e-12, mMax,resRatio,cylRadius,rPrime,zEval,thisTheta, rElec)
    print('starting electrode position ', i, ' of ', len(rElecRange))

    # Loop on Z position; but in this streamlined version, keep reval and theta constant
    # Evaluate only for y negative and 0 and reflect the voltage values to positive y positions, saving compute time
    for m, thisZ in enumerate(fp['zEval']):
        frac_done = (((i*len(fp['zEval'])) + m)*100)/(len(rElecRange) * len(fp['zEval']))
        print('# ', m, ' of ', nZ, ' z values. Approximately ', '%.2f' % frac_done, ' % complete.')
        # loop on y positions to get 2nd spatial derivative
        for j, yVal in enumerate(yVals[0:n_yeval + 1]):
            thisTheta = np.arctan(yVal / fp['reval'])
            rPrime = np.sqrt((yVal ** 2) + (fp['reval'] ** 2))  # distance of eval point from center of cylinder
            [itemp, error] = integ.quad(integ_func, fp['intStart'], fp['intEnd'], epsabs=fp['ITOL'], limit=1000,
                                        args=(fp['mMax'], resRatio, fp['cylRadius'], rPrime, thisZ, thisTheta, rElec))
            tempV = itemp / (2 * (np.pi ** 2))  # From Goldwyn eqn. 11
            voltageVals[i, j, m] = tempV
            voltageVals[i, n_ypos - (j + 1), m] = tempV  # place same value in mirror-symmetric position
            # Extra variable to make derivative calculation easy
            transVoltage[j] = tempV
            transVoltage[n_ypos - (j + 1)] = tempV

        # Calculate the second spatial derivative in the y dimension
        transVPrime = np.diff(np.diff(transVoltage)) / (y_inc ** 2)
        activationVals[i, m] = (transVPrime[n_yeval - 1])  # Value in center
        if np.isnan(activationVals[i, m]):
            print('nan value for i == ', i, " and m == ", m)

pr.disable()  # stop the profiler

# Display profiler results
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)
print(s.getvalue())

# Save the data into a single file
fp['run_duration'] = ps.total_tt  # Place run duration in field parameters
with open(fp['runOutFile'], 'wb') as combined_data:
    pickle.dump([fp, voltageVals, activationVals], combined_data)  # Save all data
combined_data.close()

nan_locs = np.argwhere(np.isnan(activationVals))
if nan_locs.size != 0:
    print('WARNING: There were NaN values in the act_vals array at locations: ', nan_locs)
    r_nans.remove_nans(fp['runOutFile'])
    print('These have been replaced by interpolated values in a new file with no_nans appended to the filename.')
else:
    print('Complete: No nans are in the file.')

if if_plot:
    # Some plots that might be helpful
    # plt.plot(rElecRange, voltageVals[:, 10], 'or')
    # plt.plot(rElecRange, activationVals[:, 0], 'ob')
    # plt.xlabel('Electrode radial position (mm)')
    # plt.ylabel('Red: voltage; Blue: activation')
    # plt.xlim(-1.0, 1.0)
    # plt.yscale('log')

    plt.figure()
    plt.xlabel('y position (mm)')
    plt.ylabel('Voltage')
    plt.plot(yVals, voltageVals[0, :, 0], '.b')
    plt.plot(yVals, voltageVals[1, :, 0], '.g')
    # plt.plot(yVals, voltageVals[2, :, 0], '.r')
    # plt.plot(yVals, voltageVals[3, :, 0], '.k')
    plt.show()
