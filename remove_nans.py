#  remove_nans.py
#  David Perkel  13 September 2022
#  This script reads a voltage table, finds nans and interpolates to remove them

import pickle
import numpy as np
from scipy import interpolate as interp


def remove_nans(f_name):
    out_fname = f_name[0:-4] + '_nonans.dat'
    with open(f_name, 'rb') as combined_data:
        data = pickle.load(combined_data)
    combined_data.close()

    fp = data[0]
    v_vals = data[1]
    act_vals = data[2]

    nan_locs = np.argwhere(np.isnan(act_vals))
    print('nanlocs[0] = ', nan_locs[0])
    for loc in nan_locs:
        print('loc is ', loc)
        nbr1 = act_vals[loc[0]-1, loc[1]]
        nbr2 = act_vals[loc[0]+1, loc[1]]
        print('nbrs: ', nbr1, nbr2)
        if np.isnan(nbr1) or np.isnan(nbr2):
            # Handle case with multiple nans together
            # Assume it's just 2 nans together
            rpos = fp['relec']
            if np.isnan(nbr1):
                # spline interpolation
                tck = interp.splrep(rpos[(loc[0] - 3): (loc[0] + 3)], act_vals[(loc[0] - 3): (loc[0] + 3), loc[1]])
                interp_val = interp.splev(loc[0], tck)
                act_vals[loc[0], loc[1]] = interp_val
            if np.isnan(nbr2):
                tempx = rpos[(loc[0] - 3): (loc[0] + 5)]
                tempy = act_vals[(loc[0] - 3): (loc[0] + 5), loc[1]]
                nan_vals = np.isnan(tempy)
                tempy[nan_vals] = 0.0
                interp_vals = interp.UnivariateSpline(tempx, tempy, w=~nan_vals)
                new_val = interp_vals(tempx[3])
                act_vals[loc[0], loc[1]] = new_val

        else:
            act_vals[loc[0], loc[1]] = np.mean([nbr1, nbr2])

    print('done, ready to save')
    # When complete, save data
    with open(out_fname, 'wb') as combined_data:
        pickle.dump([fp, v_vals, act_vals], combined_data)  # Save all data
    combined_data.close()


if __name__ == '__main__':
    filename = 'original_voltage_table_with_nans.dat'
    remove_nans(filename)
