import pandas as pd
from covid import utils
from covid import plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

# Read in data
states = pd.read_csv('data/fhi_data_daily.csv',
                     usecols=['date', 'region', 'I_cumulative'],
                     parse_dates=['date'],
                     index_col=['region', 'date'],
                     squeeze=True).sort_index()
states = states.astype('float64')

# define region and redefine first level in Series
state_name = 'Norway'
cases = states.xs(state_name).rename(f"{state_name} cases")
cases[0] = 1

# gets the original and smoothed data 
original = cases.diff()
smoothed = utils.smooth_data(original)

# Need the increment to start with one to calculate posteriors
idx_start = np.searchsorted(smoothed, 1)
smoothed = smoothed.iloc[idx_start:]
original = original.loc[smoothed.index]


# plots the original and smoothed data
# plot.smoothed_development(original, smoothed, "Norway - New Cases per Day")

# define parameters to calculate posteriors
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
gamma = 1/7

# calculate posteriors 
posteriors, log_likelihood = utils.get_posteriors(smoothed, gamma, r_t_range, sigma=.15)

# plot daily posteriors
# plot.posteriors(posteriors, 'Norway- Daily Posterior for $R_t$')

# finds the posterior intervals
hdis = utils.highest_density_interval(posteriors, percentile=0.9)
most_likely = posteriors.idxmax().rename('ML')
result = pd.concat([most_likely, hdis], axis=1)
# print(hdi.head())

# plot R_t development
start_date = '2020-02-21'
plot.plot_rt(result, start_date)

