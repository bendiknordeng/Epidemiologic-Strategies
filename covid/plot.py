import sys, os
sys.path.append(os.getcwd() + "/covid") # when main is one level above this file.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import utils
import seaborn as sns
from datetime import timedelta
import pandas as pd
import numpy as np


from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from scipy.interpolate import interp1d
from matplotlib import ticker
from matplotlib.colors import ListedColormap


color_scheme = {
    "S": '#0000ff',
    "E1": '#fbff03',
    "E2": '#ff7803',
    "A": '#bf00ff',
    "I": '#ff0000',
    "R": '#2be800',
    "D": '#000000',
    "V": '#00e8e0'
}

def seir_plot_weekly(res, start_date, labels):
    """ plots accumulated SEIR curves
    
    Parameters
        res: numpy array with shape (decision_period*horizon, #compartments)
        start_date: datetime object giving start date
        labels: labels for plotted compartments 
    """
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Weekly compartment values')
    for i in range(len(labels)):
        plt.plot(res[::, i], color=color_scheme[labels[i]], label=labels[i])
    ticks = min(len(res), 20)
    step = int(np.ceil(len(res)/ticks))
    weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    plt.ylabel("Compartment values")
    plt.xlabel("Week")
    plt.legend()
    plt.grid()
    plt.show()

def age_group_infected_plot_weekly(res, start_date, labels, R_eff, include_R=False):
    """ plots infection for different age groups per week
    
    Parameters
        res: numpy array with shape (horizon, compartments, regions, age_groups)
        start_date: datetime object giving start date
        labels: labels for plotted age_groups 
    """
    fig, ax1 = plt.subplots(figsize=(10,5))
    fig.suptitle('Weekly infected in each age group')
    lines = []
    for i, label in enumerate(labels):
        lines.append(ax1.plot(res[:, 1, i], label=label)[0])
    lines.append(ax1.plot(res.sum(axis=2)[:,1], color='r', linestyle='dashed', label="All")[0])
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Infected')
    if include_R:
        # R_eff = utils.moving_average(R_eff, 3)
        ax2 = ax1.twinx()
        lines.append(ax2.plot(R_eff[:len(res)], color='k', linestyle='dashdot', label="R_eff")[0])
        ax2.set_ylabel('R effective')

    ticks = min(len(res), 20)
    step = int(np.ceil(len(res)/ticks))
    weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    labels = [ln.get_label() for ln in lines]
    plt.legend(lines, labels)
    plt.grid()
    plt.show()

def age_group_infected_plot_weekly_cumulative(res, start_date, labels):
    """ plots cumulative infection for different age groups per week
    
    Parameters
        res: numpy array with shape (decision_period*horizon, #compartments)
        start_date: datetime object giving start date
        labels: labels for plotted age_groups 
    """
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Weekly cumulative infected in each age group')
    for i, label in enumerate(labels):
        plt.plot(np.cumsum(res[:, i]), label=label) 
    plt.plot(np.cumsum(res.sum(axis=1)), color='r', linestyle='dashed', label="All")
    ticks = min(len(res), 20)
    step = int(np.ceil(len(res)/ticks))
    weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    plt.ylabel("Infected (cumulative)")
    plt.xlabel("Week")
    plt.legend()
    plt.grid()
    plt.show()

def plot_control_measures(path, all=False):
    new_infected = utils.smooth_data(pd.Series(np.array([np.sum(s.new_infected) for s in path]).T)).values
    if all:
        c_weights_home = [s.contact_weights[0] for s in path]
        c_weights_school = [s.contact_weights[1] for s in path]
        c_weights_work = [s.contact_weights[2] for s in path]
        c_weights_public = [s.contact_weights[3] for s in path]
        weeks = [s.date.isocalendar()[1] for s in path]
        ticks = min(len(path), 20)
        step = int(np.ceil(len(path)/ticks))

        fig, ax1 = plt.subplots(figsize=(10,5))
        fig.suptitle('Control measures given infection')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('New infected')
        ln1 = ax1.plot(new_infected, color='red', linestyle='dashed', label="New infected")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Weight')
        ln2 = ax2.plot(c_weights_home, label="Home")
        ln3 = ax2.plot(c_weights_school, label="School")
        ln4 = ax2.plot(c_weights_work, label="Work")
        ln5 = ax2.plot(c_weights_public, label="Public")
        
        lines = ln1+ln2+ln3+ln4+ln5
        labels = [ln[0].get_label() for ln in [ln1, ln2, ln3, ln4, ln5]]
        plt.legend(lines, labels)
        plt.xticks(np.arange(0, len(path), step), weeks[::step])
        plt.grid()
        plt.show()
    else:
        mean_weights = np.array([s.contact_weights for s in path]).mean(axis=1)
        
        weeks = [s.date.isocalendar()[1] for s in path]
        ticks = min(len(path), 20)
        step = int(np.ceil(len(path)/ticks))

        fig, ax1 = plt.subplots(figsize=(10,5))
        fig.suptitle('Control measures given infection')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('New infected')
        ln1 = ax1.plot(new_infected, color='red', linestyle='dashed', label="New infected")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Weight')
        ln2 = ax2.plot(mean_weights, label="Mean weighting")
        
        lines = ln1+ln2
        labels = [ln[0].get_label() for ln in [ln1, ln2]]
        plt.legend(lines, labels)
        plt.xticks(np.arange(0, len(path), step), weeks[::step])
        plt.grid()
        plt.show()

def smoothed_development(original, smoothed, title):
    original.plot(title=title, c='k', linestyle=':', alpha=.5, label='Actual', legend=True, figsize=(500/72, 300/72))
    ax = smoothed.plot(label='Smoothed', legend=True, c="r")
    ax.get_figure().set_facecolor('w')
    plt.show()

def posteriors(posteriors, title):
    ax = posteriors.plot(title=title, legend=False, lw=1, c='k',alpha=.3, xlim=(0.4,6))
    ax.set_xlabel('$R_t$');
    plt.show()

def plot_rt(result):
    """ plot R_t development
    """
    fig, ax = plt.subplots(figsize=(600/72,400/72))
    ax.set_title('Real-time $R_t$')

    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[np.linspace(BELOW, MIDDLE, 25), np.linspace(MIDDLE, ABOVE, 25)])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index, values, s=30, lw=.5, c=cmap(color_mapped(values)), zorder=2)

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index), result['Low_90'].values, bounds_error=False, fill_value='extrapolate')
    highfn = interp1d(date2num(index), result['High_90'].values, bounds_error=False, fill_value='extrapolate')
    extended = pd.date_range(start=index[0], end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended, lowfn(date2num(extended)), highfn(date2num(extended)), color='k', alpha=.1, lw=0, zorder=3)
    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(-1, 5.0)
    ax.set_xlim(result.index.get_level_values('date')[0] - pd.Timedelta(days=2) , result.index.get_level_values('date')[-1]+pd.Timedelta(days=2),)
    fig.set_facecolor('w')
    fig.autofmt_xdate()
    plt.show()

def plot_heatmaps(C, weights, age_labels, fpath=""):
    """ Plots heatmaps for contact matrices

    Parameters
        C: lists of lists with contact matrices for home, work, schools, transport and leisure 
        fpath: file paths where the heat mats is saved
        weights: weights used to weight different contact matrices
    """
    matrices = C.copy()
    c_descriptions = ['Home', 'School', 'Work', 'Public', 'Combined']   
    sns.set(font_scale=1.2)
    c_combined =  np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
    matrices.append(c_combined)
    for i in range(len(matrices)):
        plt.figure(figsize = (10,7))
        sns.heatmap(np.round(matrices[i],2), annot=True, vmax=1, vmin=0, cmap="Reds", xticklabels=age_labels, yticklabels=age_labels)
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
        plt.yticks(rotation=0)
        if fpath:
            plt.savefig(fpath + c_descriptions[i])
        else:
            plt.title(c_descriptions[i])
            plt.show()

