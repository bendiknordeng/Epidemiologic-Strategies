import sys, os
sys.path.append(os.getcwd() + "/covid") # when main is one level above this file.
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as ctx 
import fiona as fi
from pyproj import CRS
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import re
from os import listdir
import imageio
from seir import SEIR
from utils import *
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap


def load_od_matrices(fpath_od):
    """ load OD-matrices representing movement between regions
    Parameters
        fpath_od: filepath to OD-matrices (.pkl file)
    Returns
        3d-array e.g (84, 256, 256)
    """
    pkl_file = open(fpath_od, 'rb')
    OD_matrices = pkl.load(pkl_file)
    pkl_file.close()
    return OD_matrices

def create_population(fpath_muncipalities_names, fpath_muncipalities_pop):
    """ load population information into dataframes
    Parameters
        fpath_muncipalities_names: file path to region names
        fpath_muncipalities_pop: file path to population data
    Returns
        (pop, befolkning), where pop is a matrix of dimensions (84, 356) and befolkning is a dataframe with region_id, region_name and population
    """
    kommunenavn = pd.read_csv(fpath_muncipalities_names, delimiter=",").drop_duplicates()
    kommunenavn.rename(columns={"Kommunenr. 2020": "kommunenummer", "Kommunenavn 2020": "kommunenavn"}, inplace=True)
    kommunenavn.kommunenummer = kommunenavn.kommunenummer.astype(int)

    # create population 
    kommunenummer_befolkning = pd.read_csv(fpath_muncipalities_pop, delimiter=";", skiprows=1)
    kommune_id = []
    for id_name in kommunenummer_befolkning.region.str.split(" "):
        kommune_id.append(int(id_name[0]))
    kommunenummer_befolkning["kommunenummer"] = kommune_id
    kommunenummer_befolkning = kommunenummer_befolkning[["kommunenummer", "Befolkning per 1.1. (personer) 2020"]].rename(columns={ "Befolkning per 1.1. (personer) 2020": "befolkning"})
    befolkning = pd.merge(kommunenummer_befolkning, kommunenavn, on='kommunenummer', sort=True)
    befolkningsarray = befolkning.befolkning.to_numpy(dtype='float64')
    pop = np.asarray([befolkningsarray for _ in range(84)])
    return pop, befolkning

def initialize_seir(config, num_regions, num_infected=50):
    """ initialize seir model 
    Parameters
        config: namedtuple with seir parameters
        num_regions: number of regions between which you travel
        num_infected: number of infected at start of simulations 
    Returns
        a SEIR-object, found in virus_sim.py
    """
    n = num_regions                     # Number of regions 
    initialInd = config.initialInd      # Initial index of region(s) infected
    initial = np.zeros(n)               # Create initial infected array
    initial[initialInd] = num_infected  # Number of infected people in each of the initial counties infected
    seir = SEIR(R0=config.R0,
                DE= config.DE* config.periods_per_day,
                DI= config.DI* config.periods_per_day,
                I0=initial,
                HospitalisationRate=config.HospitalisationRate,
                eff=config.eff,
                HospitalIters=config.HospitalIters*config.periods_per_day)
    return seir

def load_vaccination_programme(data_period, num_regions, fpath_municipalities_v):
    """ load vaccination programme i.e how vaccines should be allocated to different regions
    Parameters 
        data_period: number of periods the mobility data represents
        num_regions: number of regions between which you travel
    Returns
        a matrix of dimension (data_period, num_regions), for each time period how many vaccines should be allocated.
    """
    m = generate_vaccine_matrix(data_period, num_regions)
    write_pickle(fpath_municipalities_v, m)
    # load vaccine schedule
    pkl_file = open(fpath_municipalities_v, 'rb')
    vacc = pkl.load(pkl_file)
    pkl_file.close()
    return vacc

def simulate(seir, pop, OD_matrices, vacc, number_of_simulations=50, num_infections=50):
    """  
    Parameters
        seir:
        pop:
        OD_matrices:
        vacc:
        number_of_simulations:
        num_infections:
    Returns

    """
    alphas = [np.ones(OD_matrices.shape) for x in range(4)]  # One == no quarantene influence. Multiplied by real flow.
    iterations = 12*number_of_simulations                    # multiplied by number of periods per day (resolution of data)
    inf = num_infections                                     # Number of random infections
    return seir.seir(pop, OD_matrices, alphas, iterations, inf, vacc)

def seir_plot(res):
    """ plots the epidemiological curves
    Parameters
        res: [3D array, compartment_id]
    """
    plt.plot(res[::12, 0], color='r', label='S')
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.plot(res[::12, 5], color='m', label='V')
    plt.legend()
    plt.show()

def create_geopandas(geopandas_from_pkl, befolkning, fpath_municipalities_geo_pkl, fpath_municipalities_geo_geojson):
    """ creates geopandas dataframe used to plot 
    Parameters
        geopandas_from_pkl: Bool, True if you have a geopandas DataFrame in pickle format to read from
    Returns
        a dataframe in geopandas format, containing population and geometry information about regions.
    """
    # epsg from kartverket data
    kommune_json = pd.read_json(fpath_municipalities_geo_geojson)
    epsg_kommune = int(kommune_json["administrative_enheter.kommune"].loc["crs"]["properties"]["name"].split(":")[1]) 
    crs_kommune = CRS.from_epsg(epsg_kommune)

    # Load geojson data
    if geopandas_from_pkl: # Set to True if you have a norge_geojson.pkl in your data folder
        norge_geojson = read_pickle(fpath_municipalities_geo_pkl)
    else:
        norge_geojson = gpd.read_file(fpath_municipalities_geo_geojson, layer='administrative_enheter.kommune')
        write_pickle(fpath_municipalities_geo_pkl, norge_geojson)
    norge_geojson.kommunenummer = norge_geojson.kommunenummer.astype(int) # Ensure right epsg
       
    # merge population and geopandas 
    kommuner_geometry = pd.merge(norge_geojson, befolkning, on='kommunenummer', sort=True).reset_index().drop_duplicates(["kommunenummer"])
    kommuner_geometry.index = kommuner_geometry.kommunenummer
    kommuner_geometry = kommuner_geometry[["kommunenavn", "befolkning", "geometry"]]
    
    kommuner_geometry.crs = {'init':f'epsg:{epsg_kommune}'}
    kommuner_geometry.crs = crs_kommune
    return kommuner_geometry

def find_exposed_limits(baseline, befolkning):
    """ calculates maximum number of infected people per 100k inhabitants
    Parameters
        geopandas_from_pkl: Bool, True if you have a geopandas DataFrame in pickle format to read from
        befolkning: 
    Returns
        a dataframe in geopandas format, containing population and geometry information about regions.
    """
    df = transform_res_to__df(baseline, 'SEIRHV', befolkning.kommunenavn.to_numpy(str), befolkning.befolkning.to_numpy(int))
    df["exposed_per_100k"] = 100000*df.E/df.Population

    min_exp_val = df.exposed_per_100k.min()
    max_exp_val = df.exposed_per_100k.max()
    
    return min_exp_val, max_exp_val

def trunc_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Parameters
        cmap
        minval
        maxval=1.0
        n
    """
    new_cmap = LinearSegmentedColormap.from_list('trunc({n}, {a:.2f}, {b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def print_hospitalized_information(res):
    """ prints hospitalized information
    Parameters
        res: [3D array, compartment_id]
    """
    print("Max number of hospitalised people: ", int(res["baseline"][0][:,4].max()))
    print("Day with max hospitalised people: ", int(res["baseline"][0][:,4].argmax()/12)) # Divide by

def plot_simulation(baseline, befolkning, hosp, kommuner_geometry, path_plots):
    """
    Parameters
        baseline:
        kommuner_geometry:
    """
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
    cmap = plt.cm.get_cmap("Blues")
    blue = cmap(200)

    # Used for colorbar 
    fig, ax = plt.subplots()
    min_exp_val, max_exp_val = find_exposed_limits(baseline, befolkning)
    h = ax.imshow(np.random.uniform(low=min_exp_val, high=max_exp_val, size=(10,10)), cmap=new_cmap)


    for time_step in tqdm(range(1,250)):
        
        kommuner_geometry['exposed_per_100k'] = 100000*baseline[time_step-1, 1, :]/befolkning.befolkning.to_numpy(int)
        #plot
        fig, ax = plt.subplots(figsize=(14,14), dpi=72)
        kommuner_geometry.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
        kommuner_geometry.plot(ax=ax, column='exposed_per_100k', cmap=new_cmap, zorder=3)
        # add background
        ctx.add_basemap(ax, attribution="", source=ctx.sources.ST_TONER_LITE, zoom='auto', alpha=0.6)
        
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.axis('off')
        plt.tight_layout()

        # Add colourbar
        plt.colorbar(mappable=h)
        
        inset_ax = fig.add_axes([0.4, 0.14, 0.37, 0.27])
        inset_ax.patch.set_alpha(0.5)
        
        inset_ax.plot(baseline[:time_step, 0].sum(axis=1), label="susceptible", color=blue, ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 1].sum(axis=1), label="exposed", color='g', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 2].sum(axis=1), label="infectious", color='r', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 3].sum(axis=1), label="recovered", color='y', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(hosp[:time_step], label="hospitalised", color='purple', ls='-', lw=1.5, alpha=0.8)
        inset_ax.plot(baseline[:time_step, 5].sum(axis=1), label="vaccinated", color='m', ls='-', lw=1.5, alpha=0.8)
        
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color=blue, s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='r', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='y', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), hosp[(time_step-1)], color='purple', s=50, alpha=0.2)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 5].sum(), color='m', s=50, alpha=0.2)
        
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color=blue, s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='r', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='y', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), hosp[(time_step-1)], color='purple', s=20, alpha=0.8)
        inset_ax.scatter((time_step-1), baseline[(time_step-1), 5].sum(), color='m', s=20, alpha=0.8)
        
        inset_ax.fill_between(np.arange(0, time_step), np.maximum(baseline[:time_step, 0].sum(axis=1), \
                                                                baseline[:time_step, 3].sum(axis=1)), alpha=0.035, color='r')
        inset_ax.plot([time_step, time_step], [0, max(baseline[(time_step-1), 0].sum(), \
                                                baseline[(time_step-1), 3].sum())], ls='--', lw=0.7, alpha=0.8, color='r')
        
        inset_ax.set_ylabel('Population', size=18, alpha=1, rotation=90)
        inset_ax.set_xlabel('Days', size=18, alpha=1)
        inset_ax.yaxis.set_label_coords(-0.15, 0.55)
        inset_ax.tick_params(direction='in', size=10)
        inset_ax.set_xlim(-4, 254)
        inset_ax.set_ylim(-24000, 6024000)
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
    """ sorts a given iterable
    Parameters
        l: iterable to be sorted
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)

def create_gif(path_gif, path_plots):
    """ generate a gif
    Parameters
        fpath_municipality_gif: 
    """
    filenames = listdir(path_plots)
    filenames = sort_in_order(filenames)

    with imageio.get_writer(path_gif, mode='I', fps=4) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread(path_plots + '{}'.format(filename))
            writer.append_data(image)

