import sys, os
sys.path.append(os.getcwd() + "/covid") # when main is one level above this file.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import geopandas as gpd
# import contextily as ctx 
# from pyproj import CRS
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import re
from os import listdir
import imageio
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

def seir_plot_one_cell(history, cellid, labels):
    """ plots SEIR curves for a single region

    Parameters
        history: 3D array with shape (decision_period*horizon, #compartments, number of regions)
        cellid: index of the region to plot SEIR curves
    """
    fig = plt.figure()
    fig.suptitle('SEIAR')
    num_periods_per_day = 4
    for i in range(len(labels)):
        plt.plot(history[::num_periods_per_day, i, cellid], color=color_scheme[labels[i]], label=labels[i])  
    plt.legend()
    plt.grid()
    plt.show()

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

def age_group_infected_plot_weekly(res, start_date, labels):
    """ plots infection for different age groups per week
    
    Parameters
        res: numpy array with shape (decision_period*horizon, #compartments)
        start_date: datetime object giving start date
        labels: labels for plotted age_groups 
    """
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Weekly infected in each age group')
    for i, label in enumerate(labels):
        plt.plot(res[:, 1, i], label=label) 
    plt.plot(res.sum(axis=2)[:,1], color='r', linestyle='dashed', label="All")
    ticks = min(len(res), 20)
    step = int(np.ceil(len(res)/ticks))
    weeknumbers = [(start_date + timedelta(i*7)).isocalendar()[1] for i in range(len(res))]
    plt.xticks(np.arange(0, len(res), step), weeknumbers[::step])
    plt.ylabel("Infected")
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

def plot_control_measures(path):
    new_infected = [np.sum(s.new_infected) for s in path]
    c_weights = [s.contact_weights[0] for s in path]
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
    ln2 = ax2.plot(c_weights, label="Contact weights")
    
    lines = ln1+ln2
    labels = [ln[0].get_label() for ln in [ln1, ln2]]
    plt.legend(lines, labels)
    plt.xticks(np.arange(0, len(path), step), weeks[::step])
    plt.grid()
    plt.show()

def plot_r_effective(path):
    r_eff = [s.r_eff for s in path]
    weeks = [s.date.isocalendar()[1] for s in path]
    ticks = min(len(path), 20)
    step = int(np.ceil(len(path)/ticks))
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(r'Historic $R_{eff}$')
    plt.plot(r_eff)
    plt.xticks(np.arange(0, len(path), step), weeks[::step])
    plt.ylabel(r"$R_{eff}$")
    plt.xlabel("Week")
    plt.grid()
    plt.show()

def smoothed_development(original, smoothed, title):
    original.plot(title=title,
                c='k',
                linestyle=':',
                alpha=.5,
                label='Actual',
                legend=True,
                figsize=(500/72, 300/72))
    ax = smoothed.plot(label='Smoothed', legend=True, c="r")
    ax.get_figure().set_facecolor('w')
    plt.show()

def posteriors(posteriors, title):
    ax = posteriors.plot(title=title,
            legend=False, 
            lw=1,
            c='k',
            alpha=.3,
            xlim=(0.4,6))
    ax.set_xlabel('$R_t$');
    plt.show()

def plot_rt(result, start_date):
    """ plot R_t development
    """
    fig, ax = plt.subplots(figsize=(600/72,400/72))
    ax.set_title('Real-time $R_t$ for Norway')
        
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low_90'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High_90'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp(start_date),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp(start_date), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1), )
    fig.set_facecolor('w')
    fig.autofmt_xdate()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.show()

def plot_heatmaps(C, weights, age_labels, fpath=""):
    """ Plotes heatmaps for contact matrices

    Parameters
        C: lists of lists with contact matrices for home, work, schools, transport and leisure 
        fpath: file paths where the heat mats is saved
        weights: weights used to weight different contact matrices
    """
    matrices = C.copy()
    c_descriptions = ['Home', 'School', 'Work', 'Transport', 'Leisure', 'Combined']   
    sns.set(font_scale=1.2)
    c_combined =  np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
    matrices.append(c_combined)
    for i in range(len(matrices)):
        plt.figure(figsize = (10,7))
        sns.heatmap(matrices[i], annot=True, vmax=1, vmin=0, cmap="Reds", xticklabels=age_labels, yticklabels=age_labels)
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)
        plt.yticks(rotation=0)
        if fpath:
            plt.savefig(fpath + c_descriptions[i])
        else:
            plt.title(c_descriptions[i])
            plt.show()


def create_geopandas(geopandas_from_pkl, population, fpath_region_geo_pkl, fpath_region_geo_geojson):
    """ Creates a geopandas dataframe that stores geometry and identifying information about regions

    Parameters
        geopandas_from_pkl: Bool, True if you have a geopandas DataFrame in pickle format to read from
        population: a DataFrame with region_id, region_name and population
        fpath_region_geo_pkl: pickle file for geopandas object if norge_geojson.pkl exists in data folder
        fpath_region_geo_geojson: geojson-file to create geopandas object from
    Returns
        a dataframe in geopandas format, containing region_id, region_name, region_population and geometry information about regions.
    """
    # epsg from kartverket data
    kommune_json = pd.read_json(fpath_region_geo_geojson)
    epsg_kommune = int(kommune_json["administrative_enheter.kommune"].loc["crs"]["properties"]["name"].split(":")[1]) 
    crs_kommune = CRS.from_epsg(epsg_kommune)

    # Load geojson data
    if geopandas_from_pkl: # Set to True if you have a norge_geojson.pkl in your data folder
        norge_geojson = utils.read_pickle(fpath_region_geo_pkl)
    else:
        norge_geojson = gpd.read_file(fpath_region_geo_geojson, layer='administrative_enheter.kommune')
        utils.write_pickle(fpath_region_geo_pkl, norge_geojson)
    norge_geojson.kommunenummer = norge_geojson.kommunenummer.astype(int) # Ensure right epsg
    norge_geojson.rename(columns={"kommunenummer": "region_id"}, inplace=True)
    
    # merge population and geopandas 
    kommuner_geometry = pd.merge(norge_geojson, population, on='region_id', sort=True).reset_index().drop_duplicates(["region_id"])
    kommuner_geometry.index = kommuner_geometry.region_id
    kommuner_geometry = kommuner_geometry[["region", "population", "geometry"]]
    
    kommuner_geometry.crs = {'init':f'epsg:{epsg_kommune}'}
    kommuner_geometry.crs = crs_kommune
    return kommuner_geometry

def find_exposed_limits(baseline, population):
    """ Calculates min and max number of exposed people per 100k inhabitants

    Parameters
        baseline: resulting matrix from simulation, (#timesteps, #compartments, #regions)
        population: dataframe, containing region_id, region_name, region_population and geometry
    Returns
        min and max number of infected people per 100k inhabitants
    """
    df = utils.transform_history_to_df(0, baseline, population, 'SEAIQRDVH')

    min_exp_val = df.E_per_100k.min()
    max_exp_val = df.E_per_100k.max()
    
    return min_exp_val, max_exp_val

def find_infected_limits(baseline, population):
    """ Calculates min and max number of infected people throughout simulation period

    Parameters
        baseline: resulting matrix from simulation, (#timesteps, #compartments, #regions)
        population: dataframe, containing region_id, region_name, region_population and geometry
    Returns
        min and max number of infected people per 100k inhabitants
    """
    df = utils.transform_history_to_df(0, baseline, population, 'SEAIQRDVH')
    df = df[['timestep', 'I']].groupby(['timestep']).sum()
    min_I_val = df.I.min()
    max_I_val = df.I.max()
    
    return min_I_val, max_I_val

def trunc_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ Creates a colormap returned for plotting, based on max and min exposed individuals per 100k

    Parameters
        cmap: a base colormap that will be modified
        minval: minimum number of exposed in all regions and all time_steps of simulation
        maxval: ma number of exposed in all regions and all time_steps of simulation
        n: nuances in the colormap
    Returns
        a modified colormap
    """
    new_cmap = LinearSegmentedColormap.from_list('trunc({n}, {a:.2f}, {b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_simulation(baseline, population, hosp, kommuner_geometry, path_plots):
    """ plots pictures of a given time resolution of exposed individuals in the different regions

    Parameters
        baseline: resulting matrix from simulation, (#timesteps, #compartments, #regions)
        population: dataframe, containing region_id, region_name, region_population and geometry
        hosp: resulting array of total hospitalised individuals throughout simulation, shape (#timesteps)
        kommuner_geometry: Geopandas DataFrame containing name of regions and geometry information about polygons making the regions.
        path_plots: path to plots where all plots are saved
    """
    ncolors = 256
    ndays = len(baseline[:,0,0])+1
    # get cmap
    color_array = plt.get_cmap('Reds')(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0.3, 1, ncolors)

    map_object = LinearSegmentedColormap.from_list(name="Reds_transp", colors=color_array)

    # register the colormap object
    plt.register_cmap(cmap=map_object)

    # plot some example data
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.rand(100,100), cmap='Reds_transp')
    plt.colorbar(mappable=h)

    cmap = plt.get_cmap('Reds_transp')
    new_cmap = trunc_colormap(cmap, 0.0, .9)

    #uncomment when ctx is used
    kommuner_geometry = kommuner_geometry.to_crs(epsg=3857)  # Convert to epsg=3857 to use contextily
    west, south, east, north = kommuner_geometry.unary_union.bounds
    params = {"axes.labelcolor":"slategrey"}
    plt.rcParams.update(params)

    # Used for colorbar 
    fig, ax = plt.subplots()
    min_exp_val, max_exp_val = find_exposed_limits(baseline, population)
    
    h = ax.imshow(np.random.uniform(low=min_exp_val, high=max_exp_val, size=(10,10)), cmap=new_cmap)
    min_I_val, max_I_val = find_infected_limits(baseline, population)

    for time_step in tqdm(range(1,ndays)):
        
        kommuner_geometry['exposed_per_100k'] = 100000*baseline[time_step-1, 1, :]/population.population.to_numpy(int)
        #plot
        fig, ax = plt.subplots(figsize=(14,14), dpi=72)
        kommuner_geometry.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
        kommuner_geometry.plot(ax=ax, column='exposed_per_100k', cmap=new_cmap, zorder=3)
        # add background
        ctx.add_basemap(ax, attribution="", source=ctx.providers.Stamen.TonerLite, zoom='auto', alpha=0.6)
        
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.axis('off')
        plt.tight_layout()

        # Add colourbar
        plt.colorbar(mappable=h)

        inset_ax = fig.add_axes([0.4, 0.14, 0.37, 0.27])
        inset_ax.patch.set_alpha(0.5)

        inset_ax.plot(baseline[:time_step, 0].sum(axis=1), label="S", color='r', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 1].sum(axis=1), label="E", color='g', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 2].sum(axis=1), label="A", color='b', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 3].sum(axis=1), label="I", color='k', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 4].sum(axis=1), label="Q", color='y', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 5].sum(axis=1), label="R", color='c', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 6].sum(axis=1), label="D", color='m', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 7].sum(axis=1), label="V", color='gold', ls='-', lw=1.5, alpha=0.8)

        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color='r', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='b', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='k', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 4].sum(), color='y', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 5].sum(), color='c', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 6].sum(), color='m', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 7].sum(), color='gold', s=50, alpha=0.2)

        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color='r', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='b', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='k', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 4].sum(), color='y', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 5].sum(), color='c', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 6].sum(), color='m', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 7].sum(), color='gold', s=20, alpha=0.8)
        
        inset_ax.fill_between(np.arange(0, time_step), np.maximum(baseline[:time_step, 0].sum(axis=1), \
                                                                baseline[:time_step, 3].sum(axis=1)), alpha=0.035, color='r')
        inset_ax.plot([time_step, time_step], [0, max(baseline[(time_step-1), 0].sum(), \
                                                baseline[(time_step-1), 3].sum())], ls='--', lw=0.7, alpha=0.8, color='r')
        
        inset_ax.set_ylabel('Population', size=18, alpha=1, rotation=90)
        inset_ax.set_xlabel('Days', size=18, alpha=1)
        inset_ax.yaxis.set_label_coords(-0.15, 0.55)
        inset_ax.tick_params(direction='in', size=10)
        inset_ax.set_xlim(0, ndays)
        inset_ax.set_ylim(int(min_I_val*0.9), int(max_I_val*1.1))
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
        plt.title("Norway Covid-19 spreading on day: {}".format(time_step), fontsize=18, color= 'dimgray')
        plt.savefig(path_plots + "flows_{}.jpg".format(time_step), dpi=fig.dpi)
        plt.close()

def sort_in_order(l):
    """ Sorts a given iterable

    Parameters
        l: iterable to be sorted
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)

def create_gif(path_gif, path_plots):
    """ Generate a gif

    Parameters
        path_gif: outpath of gif
        path_plots: path where plots that will make the gif are stored
    """
    filenames = listdir(path_plots)
    filenames = sort_in_order(filenames)

    with imageio.get_writer(path_gif, mode='I', fps=4) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(path_plots + '{}'.format(filename))
            writer.append_data(image)

def plot_historical_infected(baseline, population, kommuner_geometry, path_plots):
    """ plots pictures of a given time resolution of exposed individuals in the different regions

    Parameters
        baseline: resulting matrix from simulation, (#timesteps, #compartments, #regions)
        population: dataframe, containing region_id, region_name, region_population and geometry
        kommuner_geometry: Geopandas DataFrame containing name of regions and geometry information about polygons making the regions.
        path_plots: path to plots where all plots are saved
    """

    num_iterations = 350

    ncolors = 256
    # get cmap
    color_array = plt.get_cmap('Reds')(range(ncolors))

    # change alpha values
    color_array[:, -1] = np.linspace(0.3, 1, ncolors)

    map_object = LinearSegmentedColormap.from_list(name="Reds_transp", colors=color_array)

    # register the colormap object
    plt.register_cmap(cmap=map_object)

    # plot some example data
    fig, ax = plt.subplots()
    h = ax.imshow(np.random.rand(100,100), cmap='Reds_transp')
    plt.colorbar(mappable=h)

    cmap = plt.get_cmap('Reds_transp')
    new_cmap = trunc_colormap(cmap, 0.0, .9)

    #uncomment when ctx is used
    kommuner_geometry = kommuner_geometry.to_crs(epsg=3857)  # Convert to epsg=3857 to use contextily
    west, south, east, north = kommuner_geometry.unary_union.bounds

    params = {"axes.labelcolor":"slategrey"}
    plt.rcParams.update(params)

    # Used for colorbar 
    fig, ax = plt.subplots()
    min_exp_val, max_exp_val = 0,5
    h = ax.imshow(np.random.uniform(low=min_exp_val, high=max_exp_val, size=(10,10)), cmap=new_cmap)

    for time_step in tqdm(range(1,num_iterations)):
        
        kommuner_geometry['infected_per_100k'] = 100000*baseline[time_step-1, 0, :]/population.population.to_numpy(int)
        #plot
        fig, ax = plt.subplots(figsize=(14,14), dpi=72)
        kommuner_geometry.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
        kommuner_geometry.plot(ax=ax, column='infected_per_100k', cmap=new_cmap, zorder=3)
        # add background
        ctx.add_basemap(ax, attribution="", source=ctx.providers.Stamen.TonerLite, zoom='auto', alpha=0.6)
        
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.axis('off')
        plt.tight_layout()

        # Add colourbar
        plt.colorbar(mappable=h)
        
        inset_ax = fig.add_axes([0.4, 0.14, 0.37, 0.27])
        inset_ax.patch.set_alpha(0.5)

        inset_ax.plot(baseline[:time_step, 0].sum(axis=1), label="infectious", color='r', ls='-', lw=1.5, alpha=0.8)

        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color='r', s=50, alpha=0.2)
        
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color='r', s=20, alpha=0.8)
        

        inset_ax.fill_between(np.arange(0, time_step), np.maximum(baseline[:time_step, 0].sum(axis=1), \
                                                                baseline[:time_step, 0].sum(axis=1)), alpha=0.035, color='r')
        inset_ax.plot([time_step, time_step], [0, max(baseline[(time_step-1), 0].sum(), \
                                                baseline[(time_step-1), 0].sum())], ls='--', lw=0.7, alpha=0.8, color='r')
        
        inset_ax.set_ylabel('Population', size=18, alpha=1, rotation=90)
        inset_ax.set_xlabel('Days', size=18, alpha=1)
        inset_ax.yaxis.set_label_coords(-0.15, 0.55)
        inset_ax.tick_params(direction='in', size=10)
        inset_ax.set_xlim(0, num_iterations)
        inset_ax.set_ylim(0, 100000)
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
        plt.title("Norway Covid-19 spreading on day: {}".format(time_step), fontsize=18, color= 'dimgray')
        plt.savefig(path_plots + "historical_h{}.jpg".format(time_step), dpi=fig.dpi)
        plt.close()
