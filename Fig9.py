#  Fig_fit_summary.py
#  David Perkel 4 February 2023

from common_params import *  # import common values across all models
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy.stats as stats

# Constants
label_ypos = 0.98

# Reads a summary file, and tests whether average rpos error is less than chance based on shuffling
# You need to run the inverse model for all subjects before making this figure
summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.npy'

[scenarios, rpos_summary] = np.load(summary_file_name, allow_pickle=True)
nscen = len(scenarios)
rpos_vals = []
rpos_fit_vals = []
thresh_err_summary = np.zeros((nscen, 2))
rpos_err_summary = np.zeros(nscen)
dist_corr = np.zeros(nscen)
dist_corr_p = np.zeros(nscen)

for i, scen in enumerate(scenarios):
    rpos_vals.append(rpos_summary[i][0])
    rpos_fit_vals.append(rpos_summary[i][1])

# get detailed data from the CSV summary file
dummy__ = 0.0
summary_csv_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.csv'
with open(summary_csv_file_name, mode='r') as data_file:
    entire_file = csv.reader(data_file, delimiter=',', quotechar='"')
    for row, row_data in enumerate(entire_file):
        if row == 0:  # skip header row
            pass
        else:
            [dummy__, thresh_err_summary[row-1, 0], thresh_err_summary[row-1, 1],
             rpos_err_summary[row-1], dist_corr[row-1], dist_corr_p[row-1]] = row_data

    data_file.close()

# Loop on scenarios again. Compute pairwise mean absolute position error
mean_errs = np.zeros([nscen, nscen])
median_errors = np.zeros((nscen, nscen))
corr_vals = np.zeros([nscen, nscen])
corr_p = np.zeros([nscen, nscen])

for i, scen_i in enumerate(scenarios):
    for j, scen_j in enumerate(scenarios):
        rposerrs = np.subtract(rpos_fit_vals[i], rpos_vals[j])
        mean_errs[i, j] = np.mean(np.abs(rposerrs))
        [dist_corr, dist_corr_p] = stats.pearsonr(1.0 - (rpos_fit_vals[i]), 1.0 - (rpos_vals[j]))
        corr_vals[i, j] = dist_corr
        corr_p[i, j] = dist_corr_p

# now we have the matrix. Let's plot a histogram of all the values
a = mean_errs.flatten()
mean = np.mean(a)
std = np.std(a)
diag = np.diag(mean_errs)
diag_cr = np.diag(corr_vals)

widths = [3, 1, 1]
gs_kw = dict(width_ratios=widths)
fig1, axs1 = plt.subplots(1, 3, gridspec_kw=gs_kw, figsize=(8, 4))
fig1.tight_layout(pad=2)

axs1[0].plot(thresh_err_summary[:, 0], thresh_err_summary[:, 1], 'o', color='black')
axs1[0].plot([0, 0.8], [0, 0.8], linestyle='dashed', color='black')  # line of slope 1.0
axs1[0].set_xlabel('Monopolar threshold error (dB)')
axs1[0].set_ylabel('Tripolar threshold error (dB)')
axs1[0].spines['top'].set_visible(False)
axs1[0].spines['right'].set_visible(False)
axs1[0].text(-0.17, label_ypos, 'A', size=20, weight='bold', transform=axs1[0].transAxes)
axs1[0].text(0.77, 0.7, 'y = x', size=16)

sns.swarmplot(diag, ax=axs1[1], color='black')
median = np.median(diag)
iqrbar = stats.iqr(diag)/2.0
axs1[1].set_ylabel('Distance error (mm)')
axs1[1].spines['top'].set_visible(False)
axs1[1].spines['right'].set_visible(False)
# plot median and interquartile range
axs1[1].plot([-0.3, 0.3], [median, median], 'k')  # horizonatl line to indicate median
axs1[1].plot([-0.15, 0.15], [median-iqrbar, median-iqrbar], 'k')
axs1[1].plot([-0.15, 0.15], [median+iqrbar, median+iqrbar], 'k')
axs1[1].plot([0, 0], [median-iqrbar, median+iqrbar], 'k')  # vertical line
axs1[1].set_xlim([-0.5, 0.5])
# plt.tick_params(
#     axis='x',           # changes apply to the x-axis
#     which='both',       # both major and minor ticks are affected
#     bottom=False,       # ticks along the bottom edge are off
#     top=False,          # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off

axs1[1].set_xticks([])
axs1[1].text(-0.55, label_ypos, 'B', size=20, weight='bold', transform=axs1[1].transAxes)

sns.swarmplot(diag_cr, ax=axs1[2], color='black')
median_cr = np.median(diag_cr)
iqrbar_cr = stats.iqr(diag_cr)/2.0
axs1[2].set_ylabel('Pearson\'s r')
axs1[2].spines['top'].set_visible(False)
axs1[2].spines['right'].set_visible(False)
axs1[2].plot([-0.3, 0.3], [median_cr, median_cr], 'k')  # horizonatl line to indicate median
axs1[2].plot([-0.15, 0.15], [median_cr-iqrbar_cr, median_cr-iqrbar_cr], 'k')
axs1[2].plot([-0.15, 0.15], [median_cr+iqrbar_cr, median_cr+iqrbar_cr], 'k')
axs1[2].plot([0, 0], [median_cr-iqrbar_cr, median_cr+iqrbar_cr], 'k')  # vertical line
axs1[2].set_xlim([-0.5, 0.5])
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

axs1[2].text(-0.65, label_ypos, 'C', size=20, weight='bold', transform=axs1[2].transAxes)

# Annotate with A, B, C

# Save and display
figname = INVOUTPUTDIR + 'Fig_fit_summary.pdf'
plt.savefig(figname, format='pdf', pad_inches=0.1)
plt.show()
