import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd


# load Origin-Destination matrices
pkl_file = open('data/data_municipalities/od_municipalities.pkl', 'rb') # change to your desired directory
OD_matrices = pkl.load(pkl_file)
pkl_file.close()

print(OD_matrices.shape)


kommunenavn = pd.read_csv("data/data_municipalities/Kommunenummer_navn_2020.csv", delimiter=",").drop_duplicates()
kommunenavn.rename(columns={"Kommunenr. 2020": "kommunenummer", "Kommunenavn 2020": "kommunenavn"}, inplace=True)
kommunenavn.kommunenummer = kommunenavn.kommunenummer.astype(int)

# create population 
kommunenummer_befolkning = pd.read_csv("data/data_municipalities/Folkemengde_kommuner.csv", delimiter=";", skiprows=1)
kommune_id = []
for id_name in kommunenummer_befolkning.region.str.split(" "):
    kommune_id.append(int(id_name[0]))
kommunenummer_befolkning["kommunenummer"] = kommune_id

kommunenummer_befolkning = kommunenummer_befolkning[["kommunenummer", "Befolkning per 1.1. (personer) 2020"]].rename(columns={ "Befolkning per 1.1. (personer) 2020": "befolkning"})

befolkning = pd.merge(kommunenummer_befolkning, kommunenavn, on='kommunenummer', sort=True)

befolkning.head()

befolkningsarray = befolkning.befolkning.to_numpy(dtype='float64')
pop = np.asarray([befolkningsarray for _ in range(84)])

# Set up model 
%run virus-sim.py  # Call python files in same directory

r = OD_matrices.shape[0]  # Simulation period (e.g 84)
n = pop.shape[1]          # Number of counties (e.g 11)
N = sum(befolkningsarray) # Total population (e.g 5367580)
initialInd = [0]          # Initial index of counties infected
initial = np.zeros(n)
initial[initialInd] = 50  # Number of infected people in each of the initial counties infected

model = Param(R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, I0=initial, HospitalisationRate=0.1, eff=0.95, HospitalIters=15*12) # multiply by 12 as one day consists of 12 2-hours periods 

%run div.py
v = 'data/data_municipalities/vaccines_municipalities.pkl'
m = generate_vaccine_matrix(84, 356)
write_pickle(v, m)
# load vaccine schedule
pkl_file = open('data/data_municipalities/vaccines_municipalities.pkl', 'rb')
vacc = pkl.load(pkl_file)
pkl_file.close()
print(vacc.shape)

# run simulation
%run virus-sim.py

alphas = [np.ones(OD_matrices.shape) for x in range(4)]  # One == no quarantene influence. Multiplied by real flow.
iterations = 3000                   # Number of simulations
res = {}                            # Dictionary with results for different cases 
inf = 50                            # Number of random infections
res['baseline'] = seir(model, pop, OD_matrices, alphas, iterations, inf, vacc)

# run simulation
%run virus-sim.py

seir_plot_one_cell(res['baseline'][1], 2)

# Print hospitalized information
print("Max number of hospitalised people: ", int(res["baseline"][0][:,4].max()))
print("Day with max hospitalised people: ", int(res["baseline"][0][:,4].argmax()/12)) # Divide by

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
    
seir_plot(res["baseline"][0])

# import libraries
import pandas as pd
import geopandas as gpd
# import contextily as ctx # Is not installed with current .yml file 
import fiona as fi
from pyproj import CRS

kommuner = pd.read_json('data/data_municipalities/Basisdata_0000_Norge_25833_Kommuner_GEOJSON.geojson')
epsg_kommune = int(kommuner["administrative_enheter.kommune"].loc["crs"]["properties"]["name"].split(":")[1]) # epsg for kartverket sin data
crs_kommune = CRS.from_epsg(epsg_kommune)

# Load geojson data
%run div.py

if False: # Set to False if you have a norge_geojson.pkl in your data folder

    norge_geojson = gpd.read_file("data/data_municipalities/Basisdata_0000_Norge_25833_Kommuner_GEOJSON.geojson", layer='administrative_enheter.kommune')

    write_pickle('data/data_municipalities/norge_geojson.pkl', norge_geojson)
else:
    norge_geojson = read_pickle('data/data_municipalities/norge_geojson.pkl')

    # Ensure right epsg
norge_geojson.crs = {'init':f'epsg:{epsg_kommune}'}
norge_geojson.crs = crs_kommune
norge_geojson.kommunenummer = norge_geojson.kommunenummer.astype(int)
norge_geojson.info()


kommuner_geometry = pd.merge(norge_geojson, befolkning, on='kommunenummer', sort=True).reset_index().drop_duplicates(["kommunenummer"])
kommuner_geometry.index = kommuner_geometry.kommunenummer
kommuner_geometry = kommuner_geometry[["kommunenavn", "befolkning", "geometry"]]
kommuner_geometry.head()

norge_geojson.crs

#uncomment when ctx is used

#norge_geojson_3857 = norge_geojson.to_crs(epsg=3857)  # Convert to epsg=3857 to use contextily
#west, south, east, north = norge_geojson_3857.unary_union.bounds
west, south, east, north = kommuner_geometry.unary_union.bounds

# declare baseline array storing the dynamics of the compartments 
baseline = res['baseline'][1][::12, :, :]
print(baseline.shape)
print(baseline)

# declare hopsitalisation array storing the dynamics of the hospitalised 
hosp = res['baseline'][0][::12, 4]
print(hosp.shape)
print(hosp)

%run transpose_results.py

df = transform_res_to__df(baseline, 'SEIRHV', befolkning.kommunenavn.to_numpy(str), befolkning.befolkning.to_numpy(int))
df["exposed_per_100k"] = 100000*df.E/df.Population

max_exp_ind = np.where(df.exposed_per_100k == df.exposed_per_100k.max())[0].item()
max_exp_val = df.exposed_per_100k.max()
print(max_exp_ind, max_exp_val)

# find maximum hospitalisation value to make sure the color intensities in the animation are anchored against it
#max_exp_ind = np.where(baseline[:, 1, :] == baseline[:, 1, :].max())[0].item()
#max_exp_val = baseline[:, 1, :].max()
#print(max_exp_ind, max_exp_val)

ncolors = 256
# get cmap
color_array = plt.get_cmap('Reds')(range(ncolors))
print(color_array.shape)
print(color_array)

# change alpha values
color_array[:, -1] = np.linspace(0.3, 1, ncolors)

# create colormap object
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

map_object = LinearSegmentedColormap.from_list(name="Reds_transp", colors=color_array)

# register the colormap object
plt.register_cmap(cmap=map_object)

# plot some example data
fig, ax = plt.subplots()
h = ax.imshow(np.random.rand(100,100), cmap='Reds_transp')
plt.colorbar(mappable=h)

def trunc_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list('trunc({n}, {a:.2f}, {b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('Reds_transp')
new_cmap = trunc_colormap(cmap, 0.0, .9)

# plot some example data
fig, ax = plt.subplots()
h = ax.imshow(np.random.rand(100,100), cmap=new_cmap)
plt.colorbar(mappable=h)

params = {"axes.labelcolor":"slategrey"}
plt.rcParams.update(params)
cmap = plt.cm.get_cmap("Blues")
blue = cmap(200)

from tqdm import tqdm

# Used for colorbar 
max_exp_val = baseline[:, 1, :].max()
min_exp_val = baseline[:, 1, :].min()
h = ax.imshow(np.random.uniform(low=min_exp_val, high=max_exp_val, size=(10,10)), cmap=new_cmap)

for time_step in tqdm(range(1,250)):
    
    kommuner_geometry['exposed_per_100k'] = 100000*baseline[time_step-1, 1, :]/befolkning.befolkning.to_numpy(int)
    #print(kommuner_geometry[['kommunenavn', 'exposed_per_100k']].head())
    #plot
    fig, ax = plt.subplots(figsize=(14,14), dpi=72)
    #norge_geojson_3857.loc[norge_geojson_3857.index==84, 'exposed'] = max_exp_val + 1
    kommuner_geometry.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
    kommuner_geometry.plot(ax=ax, column='exposed_per_100k', cmap=new_cmap, zorder=3)
    # add background
    # ctx.add_basemap(ax, attribution="", source=ctx.sources.ST_TONER_LITE, zoom='auto', alpha=0.6)
    
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


import re

def sort_in_order( l ):
    """ sorts a given iterable
    
    l : iterable to be sorted"""
    
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)

from os import listdir

filenames = listdir("plots/plots_municipalities/")
filenames = sort_in_order(filenames)
print(filenames)

import imageio
from tqdm import tqdm
with imageio.get_writer('gifs/gifs_municipalities/Covid_19_municipalities.gif', mode='I', fps=4) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread('plots/plots_municipalities/{}'.format(filename))
        writer.append_data(image)