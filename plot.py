import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import seaborn as sns
from datetime import timedelta
from matplotlib.dates import date2num
from matplotlib import dates as mdates
from scipy.interpolate import interp1d
from matplotlib import ticker
from matplotlib.colors import ListedColormap
# import contextily as ctx
from tqdm import tqdm
import imageio
from os import listdir
import re

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
    c_combined =  utils.generate_weighted_contact_matrix(C, weights)
    matrices.append(c_combined)
    for i in range(len(matrices)):
        plt.figure(figsize = (10,7))
        sns.heatmap(np.round(matrices[i],2), annot=True, vmax=1, vmin=0, cmap="Spectral_r", xticklabels=age_labels, yticklabels=age_labels)
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
        plt.yticks(rotation=0)
        if fpath:
            plt.savefig(fpath + c_descriptions[i])
        else:
            plt.title(c_descriptions[i])
            plt.show()

def seir_plot_weekly_several_regions(res, start_date, comps_to_plot, regions, fpath_region_names):
    """plots SEIR plots for different regions

    Args:
        res (numpy.ndarray): data accumulated across all age groups. Shape: (#periods, #compartments, #regions)
        start_date (datetime): start date for plotting to begin 
        comp_labels (list(str)): list of compartment labels to plot 
        regions (list(str)): list of region names of the regions to plot SEIR development
    """
    all_comps = {"S":0, "E1":1, "E2":2, "A":3, "I":4, "R":5, "D":6, "V":7}
    df = pd.read_csv(fpath_region_names)
    region_indices = df[df['region_name'].isin(regions)].index.tolist()
    nrows = int(np.ceil(len(regions)/4))
    ncols = min(len(regions), 4)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,5*nrows), sharex=True)
    fig.suptitle("Weekly compartment values")
    for i in range(len(regions)):
        row = i // ncols
        col = i % ncols
        ax = axs[row][col] if nrows > 1 else axs[col]
        ax.set_title(f'{regions[i].capitalize()}')
        for comp in comps_to_plot:
            ax.plot(res[:, all_comps[comp], region_indices[i]], color=color_scheme[comp], label=comp)
        weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
        ax.set_xlabel("Week")
        ax.legend()
        ax.grid()
    ticks = min(len(res), 10)
    step = int(np.ceil(len(res)/ticks))
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    plt.show()

def infection_plot_weekly_several_regions(res, start_date, regions, fpath_region_names):
    """plots infection plots for different regions

    Args:
        res (numpy.ndarray): data accumulated across all age groups. Shape: (#periods, #compartments, #regions)
        start_date (datetime): start date for plotting to begin 
        comp_labels (list(str)): list of compartment labels to plot 
        regions (list(str)): list of region names of the regions to plot infection for
    """
    df = pd.read_csv(fpath_region_names)
    region_indices = df[df['region_name'].isin(regions)].index.tolist()
    nrows = int(np.ceil(len(regions)/4))
    ncols = min(len(regions), 4)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,5*nrows), sharex=True)
    fig.suptitle("Weekly infection numbers")
    for i in range(len(regions)):
        row = i // ncols
        col = i % ncols
        ax = axs[row][col] if nrows > 1 else axs[col]
        ax.set_title(f'{regions[i].capitalize()}')
        ax.plot(res[:, region_indices[i]], label="New infected", color="tab:red")
        weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
        ax.set_xlabel("Week")
        ax.legend(loc=2)
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(res[:, region_indices[i]]), label="Cumulative total infected", color="tab:orange")
        ax2.legend(loc=4)
        ax.grid()
    ticks = min(len(res), 10)
    step = int(np.ceil(len(res)/ticks))
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    fig.tight_layout(pad=3.0)
    plt.show()
        
def find_infected_limits(res, population, per_100k):
    """finds maximum number of infected throughout the infection period
    Args:
        res (3D-array): array with compartment values. Shape: (#weeks, #compartments, "#regions")
        population (pandas.Dataframe): population for each region
        per_100k (bool): whether or not max value should be expressed as per 100k
    Returns:
        float: max value of simulation horizon
    """
    E1_index = 1
    max_E1_per_region = pd.DataFrame(res[:,E1_index,:], dtype = float).max()  # Finds max for E1
    if not per_100k:
            return max_E1_per_region.max() 
    return (max_E1_per_region / (population.population/1e5)).max()

def plot_geospatial(fpath_geospatial, res, fpath_plots, population, accumulated_compartment_plot, per_100k):
    """plots geospatial data
    Args:
        fpath_geospatial (str): filepath to json data used to load geopandas dataframe
        res (4D array): array with compartment values. Shape: (#weeks, #compartments, #regions, #age groups)
        fpath_plots (str): filepath to directory where plots will be saved
        population (pandas.Dataframe): population for each region
    """
    # plot geospatial data
    gdf = utils.generate_geopandas(population, fpath_geospatial)
    res = res.sum(axis=3) # (#weeks, #compartments, #regions)
    res_accumulated_regions = res.sum(axis=2)
    pop_factor = population.population/100000
    
    # extract bounds from gdf 
    west, south, east, north = gdf.total_bounds
    horizon = len(res_accumulated_regions)

    # Find limits for colorbar 
    v_max = find_infected_limits(res, population, per_100k)

    # make the plots 
    for time_step in tqdm(range(horizon)):
 
        # Plot geospatial data
        ix_data = 4 # S, E1, E2, A, I, R, D, V
        data_to_plot = res[time_step, ix_data,:] * pop_factor if per_100k else res[time_step, ix_data,:]
        fig, ax = plt.subplots(figsize=(14,14), dpi=72)
        gdf.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
        gdf.plot(ax=ax, column=data_to_plot, cmap='Reds', zorder=3,  legend=True, vmin=0, vmax=v_max, legend_kwds={'shrink': 0.95})
        
        # add background
        ctx.add_basemap(ax, zoom='auto', crs=3857, source=ctx.providers.Stamen.TonerLite, alpha=0.6, attribution="")
        ax.set_axis_off()
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.axis('off')
        
        # axes for compartment plot 
        inset_ax = fig.add_axes([0.4, 0.16, 0.37, 0.27]) # l:left, b:bottom, w:width, h:height
        inset_ax.patch.set_alpha(0.5)

        if accumulated_compartment_plot:
            # lines
            inset_ax.plot(res_accumulated_regions[:time_step, 0], label="S",  color=color_scheme['S'],  ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 1], label="E1", color=color_scheme['E1'], ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 2], label="E2", color=color_scheme['E2'], ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 3], label="A",  color=color_scheme['A'],  ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 4], label="I",  color=color_scheme['I'],  ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 5], label="R",  color=color_scheme['R'],  ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 6], label="D",  color=color_scheme['D'],  ls='-', lw=1.5, alpha=0.8)
            inset_ax.plot(res_accumulated_regions[:time_step, 7], label="V",  color=color_scheme['V'],  ls='-', lw=1.5, alpha=0.8)
            # circles on lines
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 0], color=color_scheme['S'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 1], color=color_scheme['E1'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 2], color=color_scheme['E2'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 3], color=color_scheme['A'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 4], color=color_scheme['I'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 5], color=color_scheme['R'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 6], color=color_scheme['D'], s=20, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 7], color=color_scheme['V'], s=20, alpha=0.8)
        else:
            inset_ax.plot(res_accumulated_regions[:time_step, 1], label="E1", color=color_scheme['E1'], ls='-', lw=1.5, alpha=0.8)
            inset_ax.scatter((time_step-1), res_accumulated_regions[time_step - 1, 1], color=color_scheme['E1'], s=20, alpha=0.8)
        
        # axes titles, label coordinates, values, font_sizes, grid, spines_colours, ticks_colours, legend, title compartment plot
        inset_ax.set_xlabel('Weeks', size=14, alpha=1, color='dimgray')
        inset_ax.tick_params(direction='in', size=10)
        inset_ax.set_xlim(-1, horizon)
        inset_ax.set_ylim(-1, 5500000) if accumulated_compartment_plot else inset_ax.set_ylim(-1, res_accumulated_regions[:, 1].max() * 1.1) 
        if accumulated_compartment_plot:
            inset_ax.yaxis.set_major_formatter(lambda x, pos: '{0:g} M'.format(x/1e6))
        inset_ax.grid(alpha=0.4)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['left'].set_color('darkslategrey')
        inset_ax.spines['bottom'].set_color('darkslategrey')
        inset_ax.tick_params(axis='x', colors='darkslategrey')
        inset_ax.tick_params(axis='y', colors='darkslategrey')
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(prop={'size':14, 'weight':'light'}, framealpha=0.5)
        plt.title("COVID-19 development in week: {}".format(time_step), fontsize=14, color='dimgray')
        plt.draw()
        plt.savefig(f"{fpath_plots}{time_step}.jpg", dpi=fig.dpi, bbox_inches = 'tight')
        plt.close()


def create_gif(fpath_gif, fpath_plots):
    """generates a gif
    Args:
        fpath_gif (str): filepath (.gif) indicating where gif will be stored
        fpath_plots (str): filepath to directory where plots is stored
    """
    def sort_in_order( l ):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanumeric_key)
    filenames = listdir(fpath_plots)
    filenames = sort_in_order(filenames)
    with imageio.get_writer(fpath_gif, mode='I', fps=4) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(fpath_plots + '{}'.format(filename))
            writer.append_data(image)