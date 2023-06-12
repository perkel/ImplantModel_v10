# A commonly called function to get full grid of survival values from
# vector of nElec survival values
import numpy as np
import getclosest


def survFull(ePos, survVec, gridPos):
    """

    :rtype: object
    """
    # ePos is an array of electrode positions, in mm
    # survVec is an array of neuron survival values at those electrode
    # positions
    # gridPos is the entire array of grid positions, typically 0:0.1:32.9

    # This function returns values interpolated within the range of the
    # electrode positions and extrapolated beyond the first and last positions
    # in a horizontal manner.

    # Special condition for ForwardModel4_2D
    if not isinstance(ePos, list):
        ELEC_BASALPOS = 30
        ESPACE = 1.1  # in mm; 'ELECTRODE' parameters must be vectors
        ePos = ELEC_BASALPOS - np.arange(16 - 1, -1, -1) * ESPACE

        # also process survival value
        survScalar = survVec
        tempSurvVec = np.ones(16) * survScalar
        survVec = tempSurvVec

    nZ = len(gridPos)
    # get location of first and last electrodes
    idxa = getclosest.getclosest(gridPos, ePos[0])
    idxb = getclosest.getclosest(gridPos, ePos[-1])
    
    survVals = np.empty(nZ)
    survVals[:] = np.nan
    nsurvtemp = np.interp(gridPos[idxa:idxb], ePos, survVec)
    survVals[0:idxa] = nsurvtemp[0]
    survVals[idxa:idxb] = nsurvtemp
    survVals[idxb:] = nsurvtemp[-1]
    
    return survVals
