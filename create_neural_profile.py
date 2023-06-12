# CIMODEL_CREATENEURALPROFILE.M
# function [neural_out,nsurvival] = cimodel_createneuralprofile(active_profile,rpos_vals)
# Create one or more neural profiles, i.e. the final neural "contribution"
# across an array of neural clusters, based on unscaled activation profiles. Called by
# thrFunction.py, but can also be evoked separately.
# 'active_profile' is of size n_z (#z-axis points) x nStim (#of stimuli to simulate).
# It should be already scaled by the applied current, but not transformed by the sidelobe
# ratio or to an absolute value. The activation sensitivity value, contained
# in 'rpos_vals.neurons.act_center' is a scalar and assumed to hold for all of the profiles.
# Note that ONLY the '.neurons' substructure of 'rpos_vals' is used in this function. Sigma,
# current level, channel, etc, are already incorporated in 'active_profile'.

import numpy as np
from scipy import special


def sigmoid_func(actr, std, x):
    retval = 0.5 * (1 + special.erf((x - actr) / (np.sqrt(2) * std)))
    return retval


def new_sigmoid(actr, std, x):  # scaled to subtract the y-intercept and still asymptote at 1.0
    y_int = sigmoid_func(actr, std, 0)
    scale_factor = 1.0 - y_int
    yval = (sigmoid_func(actr, std, x) - y_int) / scale_factor
    return yval


def create_neural_profile(active_profile, rpos_vals):
    n_z = active_profile.shape

    # Extract local values from rpos_vals structure
    act_ctr = rpos_vals['neurons']['act_ctr']
    act_std = act_ctr * rpos_vals['neurons']['act_stdrel']
    nsurvival = rpos_vals['neurons']['nsurvival']
    rlvltable = rpos_vals['neurons']['rlvl']

    # Create neural excitation profile for each input activation profile #
    neural_out = np.empty(n_z)
    neural_out[:] = np.nan

    # scale sidelobes before computing absolute value
    active_profile[active_profile < 0] = rpos_vals['neurons']['sidelobe'] * active_profile[active_profile < 0]

    if any(np.isreal(active_profile == False)):

        # treat non-viable neurons depending on the desired algorithm
        if rpos_vals['neurons']['rule'] == 'proportional':
            try:
                #  The lines commented out here are the original sigmoid activation function from Goldwyn et al. (2010)
                #  The problem is that for input of exactly zero, the output is positive, which can cause
                #  major problems in some cases.
                #  tempval1 = active_profile[:]-act_ctr
                #  tempval2 = tempval1/(np.sqrt(2)*act_std)
                #  atemp = 0.5 * (1 + sp.special.erf((tempval2 - 5)))
                atemp = new_sigmoid(act_ctr, act_std, active_profile[:])
            except BaseException:
                print('create_neural_profile: failed to calculate atemp using error function')

            ntemp = np.interp(atemp, rlvltable[0], rlvltable[1])
            if len(ntemp) == len(nsurvival):
                neural_out[:] = np.multiply(ntemp, nsurvival)
            else:
                neural_out[:] = np.multiply(ntemp, [np.nan, nsurvival, np.nan])  # if nsurvival is only 14 elements long

        else:
            raise 'Neural excitation model not recognized.'

    return neural_out
