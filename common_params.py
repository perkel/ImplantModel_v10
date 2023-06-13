# Common parameters for the implant model

import numpy as np

# Basic parameters
NELEC = 16
ELEC_BASALPOS = 30  # in mm
ESPACE = 1.1  # in mm; 'ELECTRODE' parameters must be vectors

# Neural activation parameters
THRTARG = 100.0  # threshold number of active neurons
TARG_TEXT = '_TARG' + str(round(THRTARG)) + '/'
ACTR = 100.0
ACTR_TEXT = 'ACTR' + str(round(ACTR)) + '_'
ACT_STDREL = 2.0
STD_TEXT = 'STDR' + str(ACT_STDREL)
STD_TEXT = STD_TEXT.replace('.', '_')

# File locations
sigmaVals = [0, 0.9]  # Always explore monopolar stimulation and one value of sigma for triploar

COCHLEA = {'source': 'manual', 'timestamp': [], 'radius': []}
ELECTRODES = {'source': 'manual', 'timestamp': [], 'zpos': ELEC_BASALPOS - np.arange(NELEC - 1, -1, -1) * ESPACE,
              'rpos': []}
NEURONS = {'act_ctr': ACTR, 'act_stdrel': ACT_STDREL, 'nsurvival': [], 'sidelobe': 1.0, 'neur_per_clust': 10,
           'rlvl': [], 'rule': 'proportional', 'coef': 0.0, 'power': 1.0, 'thrTarg': THRTARG}
# For COEF convex: <0 | 0.4, 0.9  linear: 0 | 1; concave: >0 | 1.0, 1.8
CHANNEL = {'source': 'manual', 'number': range(0, NELEC), 'config': 'pTP', 'sigma': 0.9, 'alpha': 0.5,
           'current': 10000000000.0}
GRID = {'r': 0.1, 'th': 0.0, 'z': np.arange(0, 33, 0.01)}  # mm
RUN_INFO = {'scenario': 'scenario', 'run_time': [], 'run_duration': 0.0}
simParams = {'cochlea': COCHLEA, 'electrodes': ELECTRODES, 'channel': CHANNEL, 'grid': GRID, 'neurons': NEURONS,
             'run_info': RUN_INFO}

nZ = len(GRID['z'])
NSURVINIT = 1.0

# Set specific scenarios to run with forward model.
# Not used by the 2D exploration tool. These are left in for convenience
# scenarios = ['Gradual80R75']
# scenarios = ['Uniform80R05']
# scenarios = ['Ramp80Rvariable1']
# scenarios = ['RampRpos_revSGradual80']
# scenarios = ['Rpos-03S0_4']
# scenarios = ['Ramp80Rvariable1']
# scenarios = ['RampRposS70']
# scenarios = ['Gradual80R-50']
# scenarios = ['RampRposS80']
# scenarios = ['RampRpos2SGradual80']
# scenarios = ['RampRposSOneHoleGradual80']

# scenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80']  # for paper figures
# scenarios = ['Gradual80R00']
# scenarios = ['ExtremeHole']
# scenarios = ['RampRposS80']
# scenarios = ['RampRposSGradual80']
# scenarios = ['CustomForECAPFigure']

# Actual subject data. For inverse model only
scenarios = ['S42', 'S43']  # paper "good fit" examples
# scenarios = ['S29', 'S56']  # paper "poor fit" examples
# all subjects with CT data
# scenarios = ['S22', 'S27', 'S29', 'S38', 'S40', 'S41', 'S42', 'S43', 'S46', 'S47', 'S49R', 'S50', 'S52', 'S53', 'S54',
#              'S55', 'S56', 'S57']

# File locations
FWD_OUT_PRFIX = 'FWD_OUTPUT/'
FWDOUTPUTDIR = FWD_OUT_PRFIX + ACTR_TEXT + STD_TEXT + TARG_TEXT
INV_OUT_PRFIX = 'INV_OUTPUT/'
INVOUTPUTDIR = INV_OUT_PRFIX + ACTR_TEXT + STD_TEXT + TARG_TEXT

FIELDTABLE = '20March2023_MedResolution_Rext250_nonans.dat'
