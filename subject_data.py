# Subject data from cochlear implant users. Thresholds and CT data showing electrode placement
import numpy as np


def subj_thr_data(subj):  # returns threshold data for a subject
    thr_data = {'thrmp_db': [], 'thrmp': [], 'thrtp_db': [], 'thrtp': [], 'thrtp_sigma': 0.9}

    if subj == 'S22':
        thr_data['thrmp_db'] = np.array([38.0, 38.0, 36.1, 36.2, 31.1, 36.2, 33.4, 37.4, 37.1, 36.9, 37.0, 37.0, 36.5,
                                         37.2, 38.0, 38.0])
        thr_data['thrtp_db'] = np.array([38.9, 34.1, 39.6, 33.4, 39.1, 33.8, 41.6, 38.8, 48.1, 45.6, 53.3,
                                         51.5, 50.1, 49.9])
    elif subj == 'S23':
        thr_data['thrmp_db'] = np.array([36.9, 36.9, 35.9, 34.6, 34.1, 33.6, 34.4, 35.1, 35.5, 35.2, 38.6, 39.0, 36.8,
                                         34.8, 35.0, 35.0])
        thr_data['thrtp_db'] = np.array([44.2, 43.5, 45.1, 45.9, 46.9, 49.9, 49.8, 44.2, 42.9, 43.6, 44.6, 44.8,
                                         42.1, 44.2])
    elif subj == 'S27':
        thr_data['thrmp_db'] = np.array([27.8, 27.8, 28.2, 28.6, 28.8, 27.7, 29.2, 30.3, 31.3, 30.1, 30.7, 31.3, 32.1,
                                         32.9, 32.9, 32.9])
        thr_data['thrtp_db'] = np.array([43.1, 43.4, 45.3, 43.5, 41.9, 41.6, 45.0, 50.3, 45.4, 44.2, 44.5, 46.2,
                                         48.2, 49.2])
    elif subj == 'S28':
        thr_data['thrmp_db'] = np.array([29.5, 29.5, 31.3, 30.6, 31.3, 30.9, 30.3, 30.7, 32.1, 31.6, 32.6, 31.5, 32.4,
                                         33.8, 36.7, 36.7])
        thr_data['thrtp_db'] = np.array([44.9, 48.5, 49.5, 44.0, 43.3, 45.0, 44.4, 47.5, 47.2, 47.7, 44.2, 45.9,
                                         46.8, 49.7])
    elif subj == 'S29':
        thr_data['thrmp_db'] = np.array([37.1, 37.1, 38.4, 38.4, 38.7, 38.4, 39.5, 40.0, 40.1, 40.1, 40.2, 41.5, 40.1,
                                         41.2, 40.4, 40.4])
        thr_data['thrtp_db'] = np.array([49.2, 51.7, 48.0, 45.8, 45.3, 41.5, 44.2, 49.0, 50.5, 45.4, 45.3, 42.2,
                                         50.3, 51.2])
    elif subj == 'S30':
        thr_data['thrmp_db'] = np.array([27.9, 27.9, 27.4, 28.9, 28.3, 29.3, 29.1, 29.6, 29.6, 29.4, 30.3, 30.3, 29.7,
                                         30.1, 31.7, 31.7])
        thr_data['thrtp_db'] = np.array([49.2, 51.7, 48.0, 45.8, 45.3, 41.5, 44.2, 49.0, 50.5, 45.4, 45.3, 42.2,
                                         50.3, 51.2])
    elif subj == 'S36':
        thr_data['thrmp_db'] = np.array([27.6, 27.6, 27.9, 28.7, 27.9, 28.4, 28.5, 28.8, 29.2, 29.1, 29.6, 27.8, 27.5,
                                         26.1, 23.5, 23.5])
        thr_data['thrtp_db'] = np.array([42.3, 44.3, 43.7, 44.2, 43.4, 44.0, 43.3, 43.0, 45.9, 42.6, 39.3, 35.8,
                                         26.9, 24.5])
    elif subj == 'S38':
        thr_data['thrmp_db'] = np.array([32.5, 32.5, 32.7, 31.1, 32.6, 30.7, 31.6, 32.1, 33.2, 32.3, 34.1, 32.8, 33.9,
                                         32.8, 32.6, 32.6])
        thr_data['thrtp_db'] = np.array([41.8, 42.9, 47.5, 48.7, 43.4, 38.5, 43.1, 46.2, 46.5, 45.8, 48.5, 49.0,
                                         45.5, 44.5])
    elif subj == 'S40':
        thr_data['thrmp_db'] = np.array([39.0, 39.0, 39.2, 39.7, 40.5, 40.3, 40.3, 40.9, 40.7, 40.7, 41.0, 41.5, 41.6,
                                         42.7, 43.5, 43.5])
        thr_data['thrtp_db'] = np.array([54.8, 55.4, 55.3, 55.2, 55.5, 52.2, 55.2, 51.3, 53.2, 55.7, 53.6, 54.0,
                                         54.2, 53.7])
    elif subj == 'S41':
        thr_data['thrmp_db'] = np.array([30.6, 30.6, 32.2, 31.3, 31.3, 33.1, 33.5, 32.7, 32.9, 33.2, 32.8, 31.9, 33.5,
                                         34.7, 36.9, 36.9])
        thr_data['thrtp_db'] = np.array([44.6, 45.2, 45.4, 44.7, 44.4, 44.2, 45.1, 45.6, 45.6, 44.4, 44.8, 44.2,
                                         47.6, 46.9])
    elif subj == 'S42':
        thr_data['thrmp_db'] = np.array([28.3, 28.3, 28.0, 28.0, 28.2, 27.7, 30.0, 29.9, 28.6, 28.1, 27.1, 27.9, 28.1,
                                         28.8, 31.0, 31.0])
        thr_data['thrtp_db'] = np.array([45.7, 45.9, 44.3, 38.5, 34.9, 34.8, 35.6, 35.5, 35.1, 32.6, 33.3, 32.9,
                                         33.3, 37.6])
    elif subj == 'S43':
        thr_data['thrmp_db'] = np.array([39.1, 39.1, 36.9, 38.4, 38.4, 37.6, 36.9, 36.4, 34.9, 33.7, 32.9, 33.3,
                                         32.5, 33.1, 36.3, 36.3])
        thr_data['thrtp_db'] = np.array([42.4, 38.2, 43.4, 45.9, 51.3, 49.8, 49.0, 46.9, 46.9, 44.7, 42.5,
                                         38.9, 41.5, 43.2])
    elif subj == 'S46':
        thr_data['thrmp_db'] = np.array([35.8, 35.8, 38.3, 37.1, 38.8, 40.1, 38.8, 40.5, 41.9, 41.0, 41.7, 41.3,
                                         40.1, 40.2, 41.0, 41.0])
        thr_data['thrtp_db'] = np.array([49.2, 49.6, 50.5, 50.7, 52.4, 52.9, 54.0, 54.3, 52.8, 51.5, 51.1, 50.8,
                                         51.6, 51.5])
    elif subj == 'S47':
        thr_data['thrmp_db'] = np.array([26.8, 26.8, 20.5, 16.7, 23.1, 23.5, 21.3, 23.3, 29.2, 20.9, 24.0, 22.8, 24.8,
                                         19.0, 19.2, 19.2])
        thr_data['thrtp_db'] = np.array([32.5, 31.6, 28.8, 32.2, 36.1, 37.6, 42.3, 43.2, 41.1, 43.5, 45.2, 40.4,
                                         38.0, 30.8])
    elif subj == 'S49R':
        thr_data['thrmp_db'] = np.array(
            [40.1, 40.1, 40.9, 40.8, 42.0, 43.0, 43.375, 43.5, 43.625, 43.7, 44.4, 44.45833333,
             44.0, 44.3, 47.4, 47.4])
        thr_data['thrtp_db'] = np.array([47.9, 50.4, 51.5, 54.7, 54.7, 52.9, 51.5, 49.1, 49, 49, 49.4, 48.8, 50, 51.1])
    elif subj == 'S50':
        thr_data['thrmp_db'] = np.array([37.9, 37.9, 36.7, 33.9, 32.7, 29.9, 32.12473841, 32.5, 33.16640507, 32.0, 23.6,
                                         33.12473841, 36.4, 38.1, 40.8, 40.8])
        thr_data['thrtp_db'] = np.array(
            [54, 52.1, 50, 47.5, 44.3, 47.3, 44.9, 41.5, 40.5, 36.9, 43.2, 39.7, 48.9, 50.7])
    elif subj == 'S52':
        thr_data['thrmp_db'] = np.array(
            [34.9, 34.9, 32.5, 31.8, 32.3, 33.9, 34.5, 34.1, 34.45833333, 34.5, 35.6, 34.29166667,
             33.0, 32.4, 33.6, 33.6])
        thr_data['thrtp_db'] = np.array(
            [39.7, 39.3, 40.4, 39.8, 40.5, 36.3, 39.5, 37.4, 40, 42.2, 41, 36.1, 37.6, 36.4])
    elif subj == 'S53':
        thr_data['thrmp_db'] = np.array([37.2, 37.2, 40.4, 40.1, 41.0, 40.5, 41.31349185, 40.8, 39.56349185, 40.3, 40.0,
                                         40.52182518, 39.4, 39.2, 40.4, 40.4])
        thr_data['thrtp_db'] = np.array([38.83, 41.9, 44, 44.92, 47.08, 49.15, 47.33, 46.98, 46.73, 47, 46.46,
                                         46.31, 44.32, 46.66])
    elif subj == 'S54':
        thr_data['thrmp_db'] = np.array([35.9, 35.9, 34.1, 36.8, 36.3, 36.9, 38.30913741, 37.1, 34.93413741, 37.0, 36.1,
                                         35.05913741, 35.1, 34.4, 33.1, 33.1])
        thr_data['thrtp_db'] = np.array([48.1, 48.5, 51.5, 53, 55.2, 54.2, 53, 41.4, 48.3, 41.1, 44.7, 42.3,
                                         40.7, 37.9])
    elif subj == 'S55':
        thr_data['thrmp_db'] = np.array([32.8, 32.8, 33.6, 33.3, 33.9, 34.1, 30.67696662, 34.8, 33.96863329, 33.7, 35.6,
                                         33.42696662, 34.8, 35.1, 35.3, 35.3])
        thr_data['thrtp_db'] = np.array([42.31835149, 39.89335149, 41.81001816, 41.46001816, 42.22668483, 41.87668483,
                                         43.84335149, 44.07668483, 44.57668483, 43.96001816, 42.21001816, 45.51001816,
                                         43.81001816, 45.36835149])
    elif subj == 'S56':
        thr_data['thrmp_db'] = np.array([34.4, 34.4, 29.0, 25.5, 30.5, 35.6, 35.39559991, 34.8, 33.60393325, 33.1, 33.9,
                                         34.47893325, 33.4, 35.9, 36.7, 36.70])
        thr_data['thrtp_db'] = np.array([47.51075867, 46.39964756, 41.95520311, 50.094092, 46.17742533, 48.26075867,
                                         47.12186978, 42.87186978, 49.51075867, 46.094092, 46.64964756, 49.37730037,
                                         46.04396703, 50.19674481])
    elif subj == 'S57':
        thr_data['thrmp_db'] = np.array([37.7, 37.7, 37.5, 37.5, 37.9, 36.3, 36.75, 36.5, 36.91666667, 35.3, 35.5,
                                         33.70833333, 32.6, 32.8, 32.9, 32.9])
        thr_data['thrtp_db'] = np.array([40.7, 42.4, 44.3, 48.9, 52.4, 50.3, 48.0, 45.4, 40.9, 46.3, 38.2, 36.1,
                                         36.4, 38.2])
    else:
        raise SystemExit('Threshold data are not available for subject ' + subj)

    return [thr_data['thrmp_db'], thr_data['thrtp_db']]


def subj_ct_data(subj):  # returns ct data for a subject
    if subj == 'S22':
        ct_vals = np.array([-0.534, -0.198, 0.200, 0.322, 0.242, -0.089, -0.315, -0.262, -0.224, -0.220, -0.336,
                            -0.283, -0.118, 0.084, 0.297, 0.427])
    elif subj == 'S23':
        ct_vals = []

    elif subj == 'S27':
        ct_vals = np.array([-0.715846995, -0.806763285, -0.868544601, -0.815384615, -0.591836735, -0.254054054,
                            -0.131868132, -0.315789474, -0.468208092, -0.034090909, 0.287234043, 0.336065574,
                            0.108225108, -0.106382979, -0.373390558, -0.753191489])
    elif subj == 'S28':
        ct_vals = []
    elif subj == 'S29':
        ct_vals = np.array([-0.748743719, -0.642458101, -0.509433962, -0.674698795, -0.65, -0.643192488,
                            -0.592920354, -0.472340426, -0.577981651, -0.400921659, -0.281553398, -0.227722772,
                            -0.145631068, -0.207373272, -0.3125, -0.417040359])
    elif subj == 'S30':
        ct_vals = []
    elif subj == 'S36':
        ct_vals = []
    elif subj == 'S38':
        ct_vals = np.array([-0.742857143, -0.637681159, -0.5, -0.317073171, -0.338235294, -0.590062112,
                            -0.447513812, -0.666666667, -0.768786127, -0.677083333, -0.664864865, -0.553398058,
                            -0.559139785, -0.484848485, -0.454545455, -0.344632768])
    elif subj == 'S40':
        ct_vals = np.array([-0.678571429, -0.579399142, -0.682403433, -0.74025974, -0.678714859, -0.661417323,
                            -0.582329317, -0.515873016, -0.566666667, -0.443983402, -0.358649789, -0.347457627,
                            -0.307359307, -0.318965517, -0.315315315, -0.426008969])
    elif subj == 'S41':
        ct_vals = np.array([-0.630769231, -0.696335079, -0.884816754, -0.834254144, -0.889447236, -0.776595745,
                            -0.778894472, -0.674418605, -0.487684729, -0.240384615, -0.086294416, 0.03125, -0.027027027,
                            -0.273684211, -0.658291457, -0.970588235])
    elif subj == 'S42':
        ct_vals = np.array([-0.763, -0.797, -0.795, -0.629, -0.018, 0.378, 0.471, 0.443, 0.380, 0.449, 0.296,
                            0.445, 0.616, 0.476, 0.549, 0.337])
    elif subj == 'S43':
        ct_vals = np.array([0.089, 0.485, 0.354, 0.205, -0.071, -0.193, -0.238, -0.315, -0.234, -0.111, 0.044, 0.350,
                            0.633, 0.694, 0.606, 0.362])
    elif subj == 'S46':
        ct_vals = np.array([-0.886597938, -0.926701571, -0.918181818, -0.913043478, -0.862068966, -0.842105263,
                            -0.727659574, -0.817391304, -0.722222222, -0.488151659, -0.264150943, -0.201877934,
                            -0.211009174, -0.351851852, -0.526315789, -0.828571429])
    elif subj == 'S47':
        ct_vals = np.array([-0.104895105, 0.04, 0.183333333, 0.574468085, 0.318181818, 0.12195122, -0.019417476,
                            -0.25, -0.402985075, -0.543147208, -0.458128079, -0.32038835, 0.004444444, 0.36318408,
                            0.801104972, 0.72972973])
    elif subj == 'S49R':
        ct_vals = np.array([0.286, 0.482, 0.13, -0.176, -0.27, -0.304, -0.34, -0.364, -0.158, 0.104, 0.368, 0.49,
                            0.61, 0.6, 0.464, 0.124])
    elif subj == 'S50':
        ct_vals = np.array([-0.686, -0.858, -0.828, -0.718, -0.732, -0.732, -0.59, -0.496, -0.248, -0.15, 0.032,
                            -0.074, -0.324, -0.824, -1, -1])
    elif subj == 'S52':
        ct_vals = np.array([0.382, 0.348, 0.564, 0.546, 0.224, 0.046, -0.056, -0.132, -0.158, 0, 0.12, 0.334,
                            0.438, 0.55, 0.582, 0.302])
    elif subj == 'S53':
        ct_vals = np.array([0.136, 0.132, 0.482, 0.37, 0.242, 0.2, 0.148, 0.112, 0.21, 0.336, 0.306, 0.388,
                            0.39, 0.486, 0.356, 0.152])
    elif subj == 'S54':
        ct_vals = np.array([0.436, 0.668, 0.414, 0.042, -0.378, -0.462, -0.558, -0.552, -0.592, -0.406, -0.144,
                            0.158, 0.556, 0.632, 0.504, 0.506])
    elif subj == 'S55':
        ct_vals = np.array([-0.808, -0.748, -0.856, -0.76, -0.71, -0.466, -0.44, -0.444, -0.346, -0.134, 0.012,
                            -0.042, 0.034, -0.06, 0.056, 0.278])
    elif subj == 'S56':
        ct_vals = np.array([-0.656, -0.456, 0.332, 0.554, 0.554, 0.548, 0.162, 0, 0.104, 0.032, 0, 0.004, 0.196,
                            0.384, 0.622, 0.65])
    elif subj == 'S57':
        ct_vals = np.array([-0.144, -0.238, -0.21, -0.37, -0.424, -0.308, -0.274, -0.442, -0.48, -0.372, -0.07,
                            0.302, 0.74, 0.74, 0.644, 0.596])

    return ct_vals
