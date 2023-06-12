# cylinder3d_makeprofile.py
# function vprofile = cylinder3d_makeprofile(fieldTable,fieldParams,simParams)

# Note that fieldTable is already embedded in simParams.grid.table, so this should be cleaned up.
# David Perkel Jan 2021


# Create the final voltage or activating profile from a standard 0-origin
# monopolar half profile defined along the z-axis, as created by CYLINDER3D and
# CREATE_VOLTAGETABLE. The process involves mirroring and translating the input
# profiles, contained in the cell array 'fieldTable' according to the instructions
# in 'simParams'.
# Like the voltage tables, the output corresponds to a current of 1 A (making the
# profile in units of Ohms), so it will still need to be scaled by the desired current.
# The radius of the model (usually 1.0) can be adapted to the desired radius by
# dividing the voltage amplitude by K and multiplying the z coordinates by K, where K
# is the ratio of desired radius to table/model radius. The 'simParams.grid.r'
# evaluation point is implemented WITHOUT this scaling; that is, it is intepreted as a
# RELATIVE radial position. Radial electrode position, on the other hand, is in
# absolute millimeter coordinates with 0 being the center of the cylinder.
# (#### This is an important distinction!! ####)
# It's up to the calling function to make sure that the simulation parameters given
# in 'simParams' match with, or can be interpolated within, the model parameters
# contained in 'fieldParams'. Note that the first two table dimensions,
# 'simParams.grid.r' and '.th', are used to narrow the dimensionality down from
# 3 to 1 based on interpolation to 'fieldParams.rspace' and the closest point to '.thspace'
# (i.e. the latter is NOT by interpolation). The z-axis uses the vector 'simParams.grid.z'
#  to linearly interpolate onto the table's z-axis specified in 'fieldParams.zspace'.
#
# REVISION HISTORY:
# 2015.06.12. Profile constructions when sigma>0 now use common values for the Region 1
# and Region 2 resistivities. This is implemented in the auxiliary function
# CYLINDER3D_INTERPOLATE. Without this change, sudden shifts in resistivity between
# neighboring electrodes could cause oddly shaped voltage profiles, because of the
# large scaling difference between the positive and negative (center and flanking)
# profile components. (Cochlear radius is already common to the center electrode, as seen
# in the 'radscale' line below.)
# 2015.08.25. There is now an option to leave 'fieldTable' empty, in which case
# the output profile of this function will be based on a voltage table made
# "on-the-fly". In this case, the table parameters are taken from 'fieldParams'.
# This operation mode has advantages for 1) flexibility and 2) using exact
# parameters rather than interpolating within a sparse parameter space.
# 2017.02.06. Output now has the z-axis along the 1st matrix dimension, so that
# the electrical field profile is a column vector.
# 2020.02.08 Translated to python by David Perkel
# 2020.08.15 Dramatically streamlined to deal with only a single 2-D activation table
# Much simpler, if a bit less powerful

import numpy as np


def cylinder3d_makeprofile(field_table, field_params, sim_params):
    # Some preliminaries
    eidx = sim_params['channel']['number']  # channel index into vector fields of 'sim_params'; must be an integer
    # Test for fatal errors    
    if sim_params['cochlea']['radius'][eidx] != field_params['cylRadius']:
        raise Exception('simulation is calling for different radius from the value in the field table')
    if field_table.size == 0:  # on-the-fly table creation; borrowed from CREATE_VOLTAGETABLE
        raise Exception('Field_table is empty')
      
    zspace = field_params['zEval']

    # Set up scaling and translation instructions for stimulation, which depends on electrode configuration #
    if sim_params['channel']['config'] == 'pTP':
        sigma = sim_params['channel']['sigma']
        alpha = sim_params['channel']['alpha']
            
        v_interp = []
        if sigma > 0.0:  # tripolar
            v_interp.append(cylinder_interpolate(field_table, field_params, sim_params, eidx))
            v_interp.append(cylinder_interpolate(field_table, field_params, sim_params, eidx))
            v_interp.append(cylinder_interpolate(field_table, field_params, sim_params, eidx))
            vloci = sim_params['electrodes']['zpos'][eidx + np.array([-1, 0, 1])]
        else:  # skip some processing steps for monopolar condition
            v_interp.append(np.zeros(field_table.shape[1]))
            v_interp.append(cylinder_interpolate(field_table, field_params, sim_params, eidx))
            v_interp.append(np.zeros(field_table.shape[1]))
            vloci = sim_params['electrodes']['zpos'][eidx + np.zeros(3, dtype=int)]
                         
        vscaling = [-sigma*(1-alpha), 1.0, -sigma*alpha]

    else:
        raise('Configuration ', sim_params['channel']['config'], ' is not recognized.')
    
    zgrid = sim_params['grid']['z']
    zextra = zspace + zspace[-1]
    zextra = zextra[:]
    n_extra = len(zextra)
    zxspace = np.concatenate([-np.flip(zextra), -np.flip(zspace), zspace, zextra])
    vprofile = np.zeros(len(zgrid))
    
    for i in range(0, len(v_interp)):  # scale, flip, and concatenate to make full profile ('zspace[0]' MUST be 0)
        if vscaling[i] != 0.0:
            vtemp = v_interp[i]
            vexpand = np.concatenate([np.zeros(n_extra), np.flip(vtemp), vtemp, np.zeros(n_extra)])
            zshifted = zxspace + vloci[i]
            vadd = np.interp(zgrid, zshifted, vexpand)
            vprofile += vadd * vscaling[i]
    
    return vprofile


def cylinder_interpolate(field_table, field_params, sim_params, reseidx):
    # Returns the voltage or activation value, interpolated to the nearest entry for 'relec'
    if isinstance(sim_params['electrodes']['rpos'], np.ndarray):  # special case for 2D version of fwd model
        relec = sim_params['electrodes']['rpos'][reseidx]
    else:
        relec = sim_params['electrodes']['rpos']
    
    # First, sanity check. Is the desired electrode position in the range?
    if (relec < min(field_params['relec'])) or (relec > max(field_params['relec'])):
        print('relec is ', relec)
        raise Exception('relec value is out of range available in voltage or activation table')

    the_shape = field_table.shape
    v_interp = np.zeros(the_shape[1])
    ncols = len(field_table[0, :])
    for i in range(0, ncols):
        v_interp[i] = np.interp(relec, field_params['relec'], field_table[:, i])
    # This is the array of activation or voltage values along the Z axis, for the interpolated relec value
    return v_interp
