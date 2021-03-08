#import the basic libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt


Param = namedtuple('Param', 'R0 DE DI I0 HospitalisationRate HospitalIters')
# I0 is the distribution of infected people at time t=0, if None then randomly choose inf number of people

def seir(par, distr, flow, alpha, iterations, inf):
    """ Simulates an epidemic
    Parameters:
        - par: parameters {
                R0: Basic reproduction number (e.g 2.4)
                DE: Incubation period (e.g 5.6 * 12)        # Needs to multiply by 12 to get one day effects
                DI: Infectious period (e.g 5.2 * 12)
                I0: In(eitial infectiouns .g initial)
                HospitalisationRate: Percentage of people that will be hospitalized (e.g 0.1)
                HospitalIters: Length of hospitalization (e.g 15*12) }
        - distr: population distribution
        - flow: OD matrices with dimensions r x n x n (i.e., 84 x 549 x 549).  flow[t mod r] is the desired OD matrix at time t. Use mod in order to only need one week of OD- matrices. 
        - alpha: strength of lock down measures/movement restriction. value of 1 - normal flow, 0 - no flow 
        - iterations: number of simulations/ duration of simulation
        - inf: number of infections
    Returns: 
        - res: matrix of shape (#iterations, #compartments" + 1(hospitalized people))
        - history: matrix with the number of subjects in each compartment [sim_it, compartment_id, num_cells]
    """
    
    r = flow.shape[0]  # Number of OD matrices - Ex. 84  
    n = flow.shape[1]  # Number of regions - Ex. 549
    N = distr[0].sum() # Total population. Assumes that N = sum(flow) 

    # Initialize compartments
    Svec = distr[0].copy()
    Evec = np.zeros(n)
    Ivec = np.zeros(n)
    Rvec = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf):
            loc = np.random.randint(n)
            if (Svec[loc] > initial[loc]):
                initial[loc] += 1.0      
    else:
        initial = par.I0
    assert ((Svec < initial).sum() == 0) # Make sure that number of suceptible > infected for every region
    
    Svec -= initial
    Ivec += initial
    
    res = np.zeros((iterations, 5))
    res[0,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
    
    # Normalise the flows and then multiply them by the alpha values. 
    # Alpha is just a conceptual measure that reduce the mobility flow
    realflow = flow.copy()
    realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]    
    realflow = alpha * realflow 
    
    history = np.zeros((iterations, 5, n)) # 5 beacuse of number of compartments
    history[0,0,:] = Svec
    history[0,1,:] = Evec
    history[0,2,:] = Ivec
    history[0,3,:] = Rvec
    
    eachIter = np.zeros(iterations + 1)
    
    # run simulation
    for iter in range(0, iterations - 1):
        realOD = realflow[iter % r]  # Stays within one week using modelo
        
        d = distr[iter % r] + 1  # At least one person in each cell. To avoid normalization problems. 

        # Partial differential equations
        newE = Svec * Ivec / d * (par.R0 / par.DI)
        newI = Evec / par.DE
        newR = Ivec / par.DI
        
        Svec -= newE
        Svec = (Svec 
               + np.matmul(Svec.reshape(1,n), realOD)
               - Svec * realOD.sum(axis=1)
                )
        Evec = Evec + newE - newI
        Evec = (Evec 
               + np.matmul(Evec.reshape(1,n), realOD)
               - Evec * realOD.sum(axis=1)
                )
                
        Ivec = Ivec + newI - newR
        Ivec = (Ivec 
               + np.matmul(Ivec.reshape(1,n), realOD)
               - Ivec * realOD.sum(axis=1)
                )
                
        Rvec += newR
        Rvec = (Rvec 
               + np.matmul(Rvec.reshape(1,n), realOD)
               - Rvec * realOD.sum(axis=1)
                )
                
        res[iter + 1,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
        eachIter[iter + 1] = newI.sum()
        res[iter + 1, 4] = eachIter[max(0, iter - par.HospitalIters) : iter].sum() * par.HospitalisationRate
        
        history[iter + 1,0,:] = Svec
        history[iter + 1,1,:] = Evec
        history[iter + 1,2,:] = Ivec
        history[iter + 1,3,:] = Rvec    
    return res, history


def seir_plot(res):
    """ Plots the epidemiological curves
    Parameters:
        res: [3D matrix, compartment_id]
    """
    plt.plot(res[::12, 0], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.legend()
    plt.show()




# load OD matrices
pkl_file = open('peder\Data\Yerevan_OD_matrices.pkl', 'rb') # change to your desired directory
OD_matrices = pickle.load(pkl_file)
pkl_file.close()
print(OD_matrices.shape)


np.set_printoptions(suppress=True, precision=3)

# load population densities
pkl_file = open('peder\Data\Yerevan_population.pkl', 'rb')
pop = pickle.load(pkl_file)
pkl_file.close()



print(pop.shape)
print(pop)

pop[13] == pop[1]

# set up model

r = OD_matrices.shape[0]
n = pop.shape[1]
N = 1000000.0

initialInd = [334, 353, 196, 445, 162, 297] # Muncipalities that is initially infected
initial = np.zeros(n)
initial[initialInd] = 50  


model = Param(R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, I0=initial, HospitalisationRate=0.1, HospitalIters=15*12)


# run simulation
# %run virus-sim.py

alpha = np.ones(OD_matrices.shape)
iterations = 3000
res = {}  # dictionary where the results for different cases will be found
inf = 50
res['baseline'], history = seir(model, pop, OD_matrices, alpha, iterations, inf)  #baseline == no quarantine

print("Max number of hospitalised people: ", int(res["baseline"][0][:,4].max()),
"\n",
"Day with max hospitalised people: ", int(res["baseline"][0][:,4].argmax()/12))
# plot result
print(res['baseline'])
seir_plot(res["baseline"][0])
print(history)


# ---------------------------- Spatial Visualization -------------------------------
"""
# import libraries
import pandas as pd
import geopandas as gpd
import contextily as ctx
from pyproj import CRS

crs = CRS.from_epsg(4326)

# load Yerevan grid file
yerevan_gdf = gpd.read_file("peder\Data\Yerevan grid shapefile\Yerevan.shp")
yerevan_gdf.crs = {'init':'epsg:4326'}
yerevan_gdf.crs = crs
yerevan_gdf.head()



# convert to crs used by contextily
yerevan_gdf_3857 = yerevan_gdf.to_crs(epsg=3857)
west, south, east, north = yerevan_gdf_3857.unary_union.bounds

# declare baseline array storing the dynamics of the compartments 
baseline = res['baseline'][1][::12, :, :]
print(baseline.shape)
print(baseline)

# declare hopsitalisation array storing the dynamics of the hospitalised 
hosp = res['baseline'][0][::12, 4]
print(hosp.shape)
print(hosp)

# find maximum hospitalisation value to make sure the color intensities in the animation are anchored against it
max_exp_ind = np.where(baseline[:, 1, :] == baseline[:, 1, :].max())[0].item()
max_exp_val = baseline[:, 1, :].max()
print(max_exp_ind, max_exp_val)

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

print("baseline dimensions: ", baseline.shape)
print("hosp dimensions: ", hosp.shape)

baseline[0, 1, :]

params = {"axes.labelcolor":"slategrey"}
plt.rcParams.update(params)
cmap = plt.cm.get_cmap("Blues")
blue = cmap(200)

from tqdm import tqdm_notebook

for time_step in tqdm_notebook(range(1,251)):
    
    yerevan_gdf_3857['exposed'] = baseline[time_step-1, 1, :]
    
    #plot
    fig, ax = plt.subplots(figsize=(14,14), dpi=72)
    yerevan_gdf_3857.loc[yerevan_gdf_3857.index==84, 'exposed'] = max_exp_val + 1
    yerevan_gdf_3857.plot(ax=ax, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=0.5, zorder=2)
    yerevan_gdf_3857.plot(ax=ax, column='exposed', cmap=new_cmap, zorder=3)
    # add background
    ctx.add_basemap(ax, attribution="", url=ctx.sources.ST_TONER_LITE, zoom='auto', alpha=0.6)
    
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.axis('off')
    plt.tight_layout()
    
    inset_ax = fig.add_axes([0.6, 0.14, 0.37, 0.27])
    inset_ax.patch.set_alpha(0.5)
    
    inset_ax.plot(baseline[:time_step, 0].sum(axis=1), label="susceptible", color=blue, ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 1].sum(axis=1), label="exposed", color='g', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 2].sum(axis=1), label="infectious", color='r', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(baseline[:time_step, 3].sum(axis=1), label="recovered", color='y', ls='-', lw=1.5, alpha=0.8)
    inset_ax.plot(hosp[:time_step], label="hospitalised", color='purple', ls='-', lw=1.5, alpha=0.8)
    
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color=blue, s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='r', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='y', s=50, alpha=0.2)
    inset_ax.scatter((time_step-1), hosp[(time_step-1)], color='purple', s=50, alpha=0.2)
    
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 0].sum(), color=blue, s=20, alpha=0.8)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 1].sum(), color='g', s=20, alpha=0.8)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 2].sum(), color='r', s=20, alpha=0.8)
    inset_ax.scatter((time_step-1), baseline[(time_step-1), 3].sum(), color='y', s=20, alpha=0.8)
    inset_ax.scatter((time_step-1), hosp[(time_step-1)], color='purple', s=20, alpha=0.8)
    
    inset_ax.fill_between(np.arange(0, time_step), np.maximum(baseline[:time_step, 0].sum(axis=1), \
                                                             baseline[:time_step, 3].sum(axis=1)), alpha=0.035, color='r')
    inset_ax.plot([time_step, time_step], [0, max(baseline[(time_step-1), 0].sum(), \
                                              baseline[(time_step-1), 3].sum())], ls='--', lw=0.7, alpha=0.8, color='r')
    
    inset_ax.set_ylabel('Population', size=18, alpha=1, rotation=90)
    inset_ax.set_xlabel('Days', size=18, alpha=1)
    inset_ax.yaxis.set_label_coords(-0.15, 0.55)
    inset_ax.tick_params(direction='in', size=10)
    inset_ax.set_xlim(-4, 254)
    inset_ax.set_ylim(-24000, 1024000)
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
    
    plt.title("Yerevan Covid-19 spreading on day: {}".format(time_step), fontsize=18, color= 'dimgray')
    
    #plt.savefig("Plots/flows_{}.jpg".format(time_step), dpi=fig.dpi)
    plt.show()

import re

def sort_in_order( l ):
    # sorts a given iterable
    
    #l : iterable to be sorted
    
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanumeric_key)

from os import listdir

filenames = listdir("Plots/")
filenames = sort_in_order(filenames)
print(filenames)

import imageio
with imageio.get_writer('Covid_19.gif', mode='I', fps=16) as writer:
    for filename in tqdm_notebook(filenames):
        image = imageio.imread('Plots/{}'.format(filename))
        writer.append_data(image)

"""