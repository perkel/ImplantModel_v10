#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:04:55 2020

@author: perkel
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

FILENAME1 = '9Sept2022_MedResolution_Rext70.dat'
FILENAME2 = '8Sept2022_MedResolution_Rext250_nonans.dat'
FILENAME3 = '9Sept2022_MedResolution_Rext6400.dat'

filenames = [FILENAME1, FILENAME2, FILENAME3]
fp = []
v_vals = []
act_vals = []
r_ext = []
for fn in filenames:
    with open(fn, 'rb') as combined_data:
        data = pickle.load(combined_data)
    combined_data.close()

    fp.append(data[0])
    v_vals.append(data[1])
    act_vals.append(data[2])
    res_idx = fn.find('Rext')
    res_end = fn.find('.')
    r_ext.append(fn[res_idx:res_end])
# First figure: comparing voltage and activation tables along z axis for radial position of 0.0

fig1, ax1 = plt.subplots()
for i, fn in enumerate(filenames):
    ax1.plot(1 - fp[i]['relec'], np.abs(v_vals[i][:, 1, 0]), marker='o')
    ax1.set(xlabel='Electrode distance (mm)', ylabel='Voltage',
           title='Voltage z = 0.0; ' + str(r_ext) + ' Ohm-cm')

fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.tick_params(axis='both', labelsize=14)
ax3 = ax2.twinx()  # second y axis
rpos = [-0.5, 0.0, 0.5]  # radial positions

# helper to calculate the fall-off over 1 mm
r_idx = 1  # hard coded -- not great, but this gives values of internal resistivity of 250 ohm-cm
falloff_dist = 1.0
z_start_idx = 0
temp0 = np.array(fp[r_idx]['zEval'])
z_falloff_idx = np.argmin(np.abs(temp0 - falloff_dist))
z_falloff_vals = np.zeros((len(rpos), 2))
for i, val in enumerate(rpos):
    rpos_idx = np.argmin(np.abs(fp[r_idx]['relec'] - val))
    if i == 0:
        ax2.plot(fp[r_idx]['zEval'], np.abs(act_vals[r_idx][rpos_idx, :]), marker='o', label='Activation')
        ax3.plot(fp[r_idx]['zEval'], np.abs(v_vals[r_idx][rpos_idx, 1, :]), marker='o', markerfacecolor='white',
                 linestyle=(0, (5, 10)), label='Voltage')

    else:
        ax2.plot(fp[r_idx]['zEval'], np.abs(act_vals[r_idx][rpos_idx, :]), marker='o')
        ax3.plot(fp[r_idx]['zEval'], np.abs(v_vals[r_idx][rpos_idx, 1, :]), marker='o', markerfacecolor='white',
             linestyle=(0, (5, 10)))

    # Calculate falloff fraction over fall-off distance
    z_falloff_vals[i, 0] = np.abs(act_vals[r_idx][rpos_idx, z_falloff_idx])/np.abs(act_vals[r_idx][rpos_idx, 0])
    z_falloff_vals[i, 1] = np.abs(v_vals[r_idx][rpos_idx, 1, z_falloff_idx]) / np.abs(v_vals[r_idx][rpos_idx, 1, 0])
    print('Act fall-off over ', falloff_dist, ' mm: for rpos: ', val, ' = ', z_falloff_vals[i, 0])
    print('Voltage fall-off over ', falloff_dist, ' mm: for rpos: ', val, ' = ', z_falloff_vals[i, 1])

# ax2.set(xlabel='Z position (mm)', ylabel='Activation', title='Activation rpos = ' + str(rpos) + ' ; Rext 250 Ohm-cm')
ax2.set_xlabel('Z position (mm)', fontsize=18)
ax2.set_ylabel('Activation', fontsize=18)
ax2.set(xlim=[-0.2, 4.2])
ax3.set_ylabel('Voltage', fontsize=18)
ax3.tick_params(axis='both', labelsize=14)
ax2.legend(prop={'size': 14})
leg2 = ax2.get_legend()
leg2.legendHandles[0].set_color('black')
ax3.legend(loc=(0.745, 0.82), prop={'size': 14})
leg3 = ax3.get_legend()
leg3.legendHandles[0].set_color('black')
plt.savefig('Fig2_ActivationTables.eps', format='eps')

# fig3, ax4 = plt.subplots()
# for i, fn in enumerate(filenames):
#     ax4.plot(1 - fp[i]['relec'], np.abs(act_vals[i][:, 0]), marker='o')
#     ax4.set(xlabel='Electrode distance (mm)', ylabel='Activation', title='Voltage z = 0.0; ' + str(r_ext) + ' Ohm-cm')
#
# fig4, ax5 = plt.subplots()
# for i, fn in enumerate(filenames):
#     max_val = np.max(np.abs(act_vals[i][:, 0]))
#     ax5.plot(1 - fp[i]['relec'], np.abs(act_vals[i][:, 0])/max_val, marker='o')
#     ax5.set(xlabel='Electrode distance (mm)', ylabel='Normalized activation',
#             title='Voltage z = 0.0; ' + str(r_ext) + ' Ohm-cm')
#
# ax5.set(xlim=[-0.1, 0.5])

# print('Activation values for z = 0.0: ', act_vals[:, 0])
# print('Calculation runtime (s): ', fp['run_duration'])
#
# fig2, ax2 = plt.subplots()  # Activation for radial position 0.0
# rpos = [-0.5, 0.0, 0.5]
# for i, val in enumerate(rpos):
#     rpos_idx = np.argmin(np.abs(fp['relec'] - val))
#     ax2.plot(fp['zEval'], act_vals[rpos_idx, :], marker='o')
#
# ax2.set(xlabel='Z position', ylabel='Activation', title='Activation rpos = ' + str(rpos) + ' ; Rext 250 Ohm-cm')
# ax2.set(xlim=[-1, 10])
#
# # Now do voltage values
# fig3, ax3 = plt.subplots()  # Multiple z positions
# zpos = [0.0, 0.2, 0.4, 0.6]
# for i, val in enumerate(zpos):
#     temparray = np.array(fp['zEval'])
#     idx = np.argwhere(temparray == val)[0]
#     yvals = np.abs(v_vals[:, 1, idx])  # the 1 represents y position == 0.0
#     ax3.plot(1 - fp['relec'], yvals, marker='o',)
# ax3.set(xlabel='Electrode distance (mm)', ylabel='Voltage', title='Voltage z = 0.0, 2.0, 10, 20; Rext 250 Ohm-cm')
#
# fig4, ax4 = plt.subplots()
# for i, val in enumerate(rpos):
#     rpos_idx = np.argmin(np.abs(fp['relec'] - val))
#     ax4.plot(fp['zEval'], np.abs(v_vals[rpos_idx, 1, :]), marker='o')
# ax4.set(xlabel='Z position', ylabel='Voltage', title='Voltage rpos = ' + str(rpos) + ' ; Rext 250 Ohm-cm')
# ax4.set(xlim=[-1, 10])
#
# fig5, ax5 = plt.subplots()
#
# ax5.contourf(fp['zEval'], fp['relec'], act_vals, np.arange(0, 10, 1), cmap='hot')
plt.show()
