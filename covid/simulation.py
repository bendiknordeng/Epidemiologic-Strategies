import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
# import contextily as ctx # Is not installed with current .yml file 
import fiona as fi
from pyproj import CRS
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import re
from os import listdir
import imageio
from virus_sim import SEIR
from utils import *
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap


def load_od_matrices():
    """ load OD-matrices representing movement between regions
    Paramters:
    Returns:
        Matrices with dimensions (num_time_steps, num_regions, num_regions)
    """
    filepath = 'data/data_municipalities/od_municipalities.pkl'
    pkl_file = open(filepath, 'rb') # change to your desired directory
    OD_matrices = pkl.load(pkl_file)
    pkl_file.close()
    print(OD_matrices.shape)
    return OD_matrices

def create_population():
    """ load population information into dataframes
    Paramters:
    Returns:
        (pop, befolkning), where pop is a matrix of dimensions (84, 356) and befolkning is a dataframe with region_id, region_name and population
    """
    kommunenavn_filepath = "data/data_municipalities/Kommunenummer_navn_2020.csv"
    kommunenavn = pd.read_csv(kommunenavn_filepath, delimiter=",").drop_duplicates()
    kommunenavn.rename(columns={"Kommunenr. 2020": "kommunenummer", "Kommunenavn 2020": "kommunenavn"}, inplace=True)
    kommunenavn.kommunenummer = kommunenavn.kommunenummer.astype(int)

    # create population 
    befolkning_filepath = "data/data_municipalities/Folkemengde_kommuner.csv"
    kommunenummer_befolkning = pd.read_csv(befolkning_filepath, delimiter=";", skiprows=1)
    kommune_id = []
    for id_name in kommunenummer_befolkning.region.str.split(" "):
        kommune_id.append(int(id_name[0]))
    kommunenummer_befolkning["kommunenummer"] = kommune_id
    kommunenummer_befolkning = kommunenummer_befolkning[["kommunenummer", "Befolkning per 1.1. (personer) 2020"]].rename(columns={ "Befolkning per 1.1. (personer) 2020": "befolkning"})
    befolkning = pd.merge(kommunenummer_befolkning, kommunenavn, on='kommunenummer', sort=True)
    befolkningsarray = befolkning.befolkning.to_numpy(dtype='float64')
    pop = np.asarray([befolkningsarray for _ in range(84)])
    return pop, befolkning

def initialize_seir(config, data_period, num_regions, tot_population, num_infected=50):
    """ initialize seir model 
    Paramters: 
        data_period: number of periods the mobility data represents
        num_regions: number of regions between which you travel
        tot_population: the total number of inhabitants in the different regions
        num_infected: number of infected at start of simulations 
    Returns:
        a SEIR-object, found in virus_sim.py
    """
    r = data_period             # Simulation period 
    n = num_regions             # Number of regions 
    N = tot_population          # Total population 
    initialInd = config.initialInd            # Initial index of region(s) infected
    initial = np.zeros(n)       # Create initial infected array
    initial[initialInd] = num_infected  # Number of infected people in each of the initial counties infected
    seir = SEIR(R0=config.R0,
                DE= config.DE* config.periods_per_day,
                DI= config.DI* config.periods_per_day,
                I0=initial,
                HospitalisationRate=config.HospitalisationRate,
                eff=config.eff,
                HospitalIters=config.HospitalIters*config.periods_per_day)
    return seir

def load_vaccination_programme(data_period, num_regions):
    """ load vaccination programme, e.g. how vaccines should be allocated to different regions
    Paramters: 
        data_period: number of periods the mobility data represents
        num_regions: number of regions between which you travel
    Returns:
        a matrix of dimension (data_period, num_regions), for each time period how many vaccines should be allocated.
    """
    
    v = 'data/data_municipalities/vaccines_municipalities.pkl'
    m = generate_vaccine_matrix(data_period, num_regions)
    write_pickle(v, m)
    # load vaccine schedule
    pkl_file = open('data/data_municipalities/vaccines_municipalities.pkl', 'rb')
    vacc = pkl.load(pkl_file)
    pkl_file.close()
    print(vacc.shape)
    return vacc

def simulate(seir, pop, OD_matrices, vacc, number_of_simulations=250, num_infections=50):
    alphas = [np.ones(OD_matrices.shape) for x in range(4)]  # One == no quarantene influence. Multiplied by real flow.
    iterations = 12*number_of_simulations                    # multiplied by number of periods per day (resolution of data)
    inf = num_infections                                     # Number of random infections
    return seir.seir(pop, OD_matrices, alphas, iterations, inf, vacc)

def seir_plot(res):
    """ Plots the epidemiological curves
    Parameters:
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

def create_geopandas(geopandas_from_pkl=True):
    """ Creates geopandas dataframe used to plot 
    Parameters:
        geopandas_from_pkl: Bool, True if you have a geopandas DataFrame in pickle format to read from
    Returns:
        a dataframe in geopandas format, containing population and geometry information about regions.
    """

    geojson_filepath = 'data/data_municipalities/Basisdata_0000_Norge_25833_Kommuner_GEOJSON.geojson'
    # epsg for kartverket sin data
    kommune_json = pd.read_json(geojson_filepath)
    epsg_kommune = int(kommune_json["administrative_enheter.kommune"].loc["crs"]["properties"]["name"].split(":")[1]) 
    crs_kommune = CRS.from_epsg(epsg_kommune)

    # Load geojson data
    if geopandas_from_pkl: # Set to True if you have a norge_geojson.pkl in your data folder
        norge_geojson = read_pickle('data/data_municipalities/norge_geojson.pkl')
    else:
        norge_geojson = gpd.read_file("data/data_municipalities/Basisdata_0000_Norge_25833_Kommuner_GEOJSON.geojson", layer='administrative_enheter.kommune')
        write_pickle('data/data_municipalities/norge_geojson.pkl', norge_geojson)
    norge_geojson.kommunenummer = norge_geojson.kommunenummer.astype(int)
    # Ensure right epsg
    

    kommuner_geometry = pd.merge(norge_geojson, befolkning, on='kommunenummer', sort=True).reset_index().drop_duplicates(["kommunenummer"])
    kommuner_geometry.index = kommuner_geometry.kommunenummer
    kommuner_geometry = kommuner_geometry[["kommunenavn", "befolkning", "geometry"]]
    
    kommuner_geometry.crs = {'init':f'epsg:{epsg_kommune}'}
    kommuner_geometry.crs = crs_kommune
    print(kommuner_geometry.crs)

    return kommuner_geometry

def find_max_exposed(baseline, befolkning):
    """ Calculates maximum number of infected people per 100k inhabitants
    Parameters:
        geopandas_from_pkl: Bool, True if you have a geopandas DataFrame in pickle format to read from
    Returns:
        a dataframe in geopandas format, containing population and geometry information about regions.
    """
    df = transform_res_to__df(baseline, 'SEIRHV', befolkning.kommunenavn.to_numpy(str), befolkning.befolkning.to_numpy(int))
    df["exposed_per_100k"] = 100000*df.E/df.Population

    max_exp_ind = np.where(df.exposed_per_100k == df.exposed_per_100k.max())[0].item()
    max_exp_val = df.exposed_per_100k.max()
    return max_exp_ind, max_exp_val

def trunc_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list('trunc({n}, {a:.2f}, {b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_simulation(kommuner_geometry):
    ncolors = 256
    # get cmap
    color_array = plt.get_cmap('Reds')(range(ncolors))
    print(color_array.shape)
    print(color_array)

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
    max_exp_val = baseline[:, 1, :].max()
    min_exp_val = baseline[:, 1, :].min()
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
        plt.savefig("plots/plots_municipalities/flows_{}.jpg".format(time_step), dpi=fig.dpi)
        plt.close()

def sort_in_order(l):
    # sorts a given iterable
    #l : iterable to be sorted

    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)


def create_gif():
    filenames = listdir("plots/plots_municipalities/")
    filenames = sort_in_order(filenames)

    with imageio.get_writer('gifs/gifs_municipalities/Covid_19_municipalities.gif', mode='I', fps=4) as writer:
        for filename in tqdm(filenames):
            image = imageio.imread('plots/plots_municipalities/{}'.format(filename))
            writer.append_data(image)

