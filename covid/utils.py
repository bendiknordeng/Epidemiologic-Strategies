import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import ast
from collections import namedtuple


def create_named_tuple(filepath):
    """ generate a namedtuple from a txt file
    Parameters
        filepath: file path to .txt file
    Returns
        A namedtuple representing each path needed for system execution
    """
    file = open(filepath, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return namedtuple('_', dictionary.keys())(**dictionary)

def generate_od_matrix(num_time_steps, num_regions):
    """ generate an OD-matrix used for illustrative purposes only
    Parameters
        num_regions: number of regions e.g 356
        num_time_steps: number of time periods e.g 84
    Returns
        An OD-matrix with dimensions (num_time_steps, num_regions, num_regions) with 0.8 on its diagonal and 0.1 on cells next to diagonal 
    """
    a = []
    for m in range(num_time_steps):
        l = [[0 for x in range(num_regions)] for x in range(num_regions)] 
        for i in range(0, num_regions):
            for j in range(0, num_regions):
                if i == j:
                    l[i][j] = 0.8
                elif j == i+1:
                    l[i][j] = 0.1
                elif j == i-1:
                    l[i][j] = 0.1
        a.append(l)
    return np.array(a)

def write_pickle(filepath, arr):
    """ writes an array to file as a pickle
    Parameters
        filepath: string file path
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ read pickle
    Parameters
        filepath: string file path
    Returns
        arr: array that is read from filepath 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)

def transform_history_to_df(time_step, history, population, column_names):
    """ transforms a 3D array that is the result from SEIR modelling to a pandas dataframe
    Parameters
        time_step: integer used to indicate time step in the simulation
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        population: DataFrame with region_id, region_namntse, and population (quantity)
        coloumn_names: string that represents the column names (e.g 'SEIRHQ'). 
    Returns
        df: dataframe that represents the 3D matrix 
    """
    A = history.transpose(0,2,1)
    (a,b,c) = A.shape 
    B = A.reshape(-1,c) 
    df = pd.DataFrame(B, columns=list(column_names))
    df['timestep'] = np.floor_divide(df.index.values, b) + time_step
    df['region_id'] = np.tile(np.array(population.region_id), a)
    df['region_name'] = np.tile(np.array(population.region), a)
    df['region_population'] = np.tile(np.array(population.population), a)
    df['E_per_100k'] = 100000*df.E/df.region_population
    return df[['timestep', 'region_id', 'region_name', 'region_population'] + list(column_names) + ['E_per_100k']]

def transform_df_to_history(df, column_names):
    """ transforms a data frame to 3D matrix that is the result from SEIR modelling
    Parameters
    df:  dataframe to b
    coloumn_names: string that represents the column names (e.g 'SEIRHQ'). 
    region_names: list with strings that represents all the regions. Should have the same length as res[2]
    Returns
    3D-matrix used for plotting geospatial and SEIR development
    """
    l = []
    df = df[list(column_names)]
    for i in range(0, len(df), 356):
        l.append(np.transpose(df.iloc[i:i+356].to_numpy()))
    return np.array(l)

def seir_plot_one_cell(history, cellid):
    """ plots SEIR development for a single region
    Parameters
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        cellid: index to the region to plot in the res 3d-array
    """
    num_periods_per_day = 4
    plt.plot(history[::num_periods_per_day, 0, cellid], color='r', label='S')  
    plt.plot(history[::num_periods_per_day, 1, cellid], color='g', label='E')
    plt.plot(history[::num_periods_per_day, 2, cellid], color='b', label='I')
    plt.plot(history[::num_periods_per_day, 3, cellid], color='y', label='R')
    plt.plot(history[::num_periods_per_day, 4, cellid], color='c', label='H')
    plt.plot(history[::num_periods_per_day, 5, cellid], color='m', label='V')
    plt.legend()
    plt.show()

def seir_plot(res):
    """ plots accumulated SEIR development
    Parameters
        res: 3D array with shape (number of time steps, number of compartments)
    """
    num_periods_per_day = 4
    plt.plot(res[::num_periods_per_day, 0], color='r', label='S') 
    plt.plot(res[::num_periods_per_day, 1], color='g', label='E')
    plt.plot(res[::num_periods_per_day, 2], color='b', label='I')
    plt.plot(res[::num_periods_per_day, 3], color='y', label='R')
    plt.plot(res[::num_periods_per_day, 4], color='c', label='H')
    plt.plot(res[::num_periods_per_day, 5], color='m', label='V')
    plt.legend()
    plt.show()

def res_from_hist(history):
    """ returns res matrix from history matrix
    Parameters
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
    Returns

    """
    return history.sum(axis=2)


