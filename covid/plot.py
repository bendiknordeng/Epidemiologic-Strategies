import sys, os
sys.path.append(os.getcwd() + "/covid") # when main is one level above this file.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import utils
import seaborn as sns
from datetime import timedelta
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from scipy.interpolate import interp1d
from matplotlib import ticker
from matplotlib.colors import ListedColormap
import contextily as ctx
from tqdm import tqdm

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

def plot_spatial(gdf, res):
    """[summary]

    Args:
        gdf ([type]): [description]
        res_accumulated_regions ([type]): [description]
    """

    res_accumulated_regions = res.sum(axis=2)

    # extract bounds from gdf 
    west, south, east, north = gdf.total_bounds

    # make the plots 
    for time_step in tqdm(range(len(res_accumulated_regions))):
 
        # Plot values on map
        ix_data = 4 # S, E1, E2, A, I, R, D, V
        data_to_plot = res[time_step, ix_data,:]
        
        # add axis for spatial plot
        fig, ax = plt.subplots(figsize=(14,14), dpi=72)
        gdf.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=3)
        gdf.plot(ax=ax, column=data_to_plot, zorder=3)
        
        # add background
        ctx.add_basemap(ax, zoom='auto', crs=3857, source=ctx.providers.Stamen.TonerLite, alpha=0.6, attribution="")
        ax.set_axis_off()
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.axis('off')
        plt.tight_layout()
        
        # axes for SEIR plot 
        inset_ax = fig.add_axes([0.6, 0.14, 0.37, 0.27])
        inset_ax.patch.set_alpha(0.5)

        # lines
        inset_ax.plot(res_accumulated_regions[:time_step, 0], label="S",  ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 1], label="E1", ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 2], label="E2", ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 3], label="A",  ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 4], label="I",  ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 5], label="R",  ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 6], label="D",  ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(res_accumulated_regions[:time_step, 7], label="V",  ls='-', lw=1.5, alpha=0.8)

        # fots on line
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 0], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 1], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 2], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 3], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 4], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 5], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 6], s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), res_accumulated_regions[time_step, 7], s=20, alpha=0.8)

        # Shaded area and vertical dotted line between S and I curves in SEIR plot 
        #inset_ax.fill_between(np.arange(0, time_step), res_accumulated_regions[:time_step, 0].sum(axis=1), res_accumulated_regions[:time_step, 3].sum(axis=1), alpha=0.035, color='r')
        #inset_ax.plot([time_step, time_step], [0, max(res_accumulated_regions[(time_step-1), 0].sum(), res_accumulated_regions[(time_step-1), 3].sum())], ls='--', lw=0.7, alpha=0.8, color='r')
        
        # axes titles, label coordinates, values, font_sizes, grid, spines_colours, ticks_colurs, legend, title for SEIR plot
        inset_ax.set_ylabel('Population', size=14, alpha=1, rotation=90)
        inset_ax.set_xlabel('Weeks', size=14, alpha=1)
        inset_ax.yaxis.set_label_coords(-0.15, 0.55)
        inset_ax.tick_params(direction='in', size=10)
        inset_ax.set_xlim(-4, num_weeks)
        inset_ax.set_ylim(-24000, 5500000)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        inset_ax.grid(alpha=0.4)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['left'].set_color('darkslategrey')
        inset_ax.spines['bottom'].set_color('darkslategrey')
        inset_ax.tick_params(axis='x', colors='darkslategrey')
        inset_ax.tick_params(axis='y', colors='darkslategrey')
        plt.legend(prop={'size':14, 'weight':'light'}, framealpha=0.5)
        plt.title("COVID-19 development in week: {}".format(time_step), fontsize=18, color= 'dimgray')

        plt.savefig("plots/flows_{}.jpg".format(time_step), dpi=fig.dpi)
        plt.clf()