# getclosest.py  By Stephen Bierer in Matlab
# Translated to python by David J. Perkel
# function idx = getclosest(actual,desired)
# Function to find the closest element of one vector
# to the elements of another vector. 'actual' is the original data,
# in numpy array form. 'desired' is the point or set of points (vector
# or scalar) to test against 'actual'. The output 'idx' is the index
# of 'actual' elements that lie closest to each 'desired'
# element; it is the same size as 'desired'.
# If two desired points are equally close, the first will be chosen (as in the MIN function).
# e.g.
# actual = [0 10 20 30 40 50];
# desired = [-2 0 9 11 20 21 26 165];
# idx = getclosest(actual,desired);
#
# idx = [  1     1     2     2     3     3     4     6  ];

import numpy as np


def getclosest(actual, desired):

    if isinstance(actual, list):
        if actual.size == 0:
            idx = []
            return idx

    if not isinstance(desired, list):
        tempval = np.abs(actual - desired)
        tempmin = np.amin(tempval)
        temp3 = (tempval == tempmin)
        idx = int(np.where(temp3)[0])
    else:
        idx = []
        for i in range(0, desired.size):
            tempval = np.abs(actual-desired[i])
            tempmin = np.amin(tempval)
            idx.append(np.where(tempval == tempmin))

    return idx
