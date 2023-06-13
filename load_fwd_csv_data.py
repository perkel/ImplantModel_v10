# load_fwd_csv_data.py
#  Utility function to load threshold, CT, and potentially other data from a
#  spreadsheet.
#  Translated to Python 3 by David Perkel December 2020

import csv
import numpy as np


def load_fwd_csv_data(loadfile):
    # file format
    # SURVCOL = 1  # unused in this code
    # RPOSCOL = 2  # unused in this code
    thr_mp_col = 3
    thr_tp_col = 4

    thr_data = {'thrmp_db': [], 'thrmp': [], 'thrtp_db': [], 'thrtp': [], 'thrtp_sigma': 0.9}
    ct_data = {'stdiameter': [], 'scala': [], 'elecdist': [], 'espace': 1.1, 'type': [], 'insrt_base': [],
               'insert_apex': []}

    radius = 1.0

    # Load the data
    with open(loadfile, mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
        for row in data_reader:
            if row[thr_mp_col] == 'nan':
                thr_data['thrmp_db'].append(np.nan)
            else:
                thr_data['thrmp_db'].append(float(np.fromstring(row[thr_mp_col], sep=' ')))
            thr_data['thrmp'].append(np.power(10.0, (thr_data['thrmp_db'][-1] / 20)))
            if row[thr_tp_col] == 'nan':
                thr_data['thrtp_db'].append(np.nan)
            else:
                thr_data['thrtp_db'].append(float(np.fromstring(row[thr_tp_col], sep=' ')))
            thr_data['thrtp'].append(np.power(10.0, (thr_data['thrtp_db'][-1] / 20)))

    n_elec = len(thr_data['thrmp_db'])
    ct_data['stdiameter'] = radius * 2.0 * (np.zeros(n_elec) + 1.0)
    return [thr_data, ct_data]
