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

def generate_dummy_od_matrix(num_time_steps, num_regions):
    """ generate an OD-matrix used for illustrative purposes only

    Parameters
        num_regions: int indicating number of regions e.g 356
        num_time_steps: int indicating number of time periods e.g 28
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

def generate_ssb_od_matrix(num_time_steps, population, fpath_muncipalities_commute):
    """ generate an OD-matrix used for illustrative purposes only

    Parameters
        num_regions: int indicating number of regions e.g 356
        num_time_steps: int indicating number of time periods e.g 28
    Returns
        An OD-matrix with dimensions (num_time_steps, num_regions, num_regions) with 0.8 on its diagonal and 0.1 on cells next to diagonal 
    """

    df = pd.read_csv(fpath_muncipalities_commute, usecols=[1,2,3])
    df['from'] = df['from'].str.lstrip("municip municip0").astype(int)
    df['to'] = df['to'].str.lstrip("municip municip0").astype(int)
    decision_period = 28
    region_id = population.region_id.to_numpy()
    od = np.zeros((decision_period, len(region_id), len(region_id)))

    morning_travel = np.zeros((len(region_id), len(region_id)))
    for id in region_id:
        i = np.where(region_id == id)
        filtered = df.where(df["from"] == id).dropna()
        to, n = filtered['to'].to_numpy(int), filtered['n'].to_numpy()
        for k in range(len(to)):
            j = np.where(region_id == to[k])
            morning_travel[i,j] = n[k]
    afternoon_travel = morning_travel.T

    for i in range(num_time_steps):
        if (i-1)%4 == 0:
            print(i)
            od[i] = morning_travel
        elif (i-3)%4 == 0:
            od[i] = afternoon_travel
    return od

def write_pickle(filepath, arr):
    """ writes an array to file as a pickle

    Parameters
        filepath: string file path
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ read pickle and returns an array

    Parameters
        filepath: string file path
    Returns
        arr: array that is read from file path 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)

def transform_history_to_df(time_step, history, population, column_names):
    """ transforms a 3D array that is the result from SEIR modelling to a pandas dataframe

    Parameters
        time_step: integer used to indicate the current time step in the simulation
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        population: DataFrame with region_id, region_names, and population (quantity)
        coloumn_names: string that represents the column names e.g 'SEIRHQ'. 
    Returns
        df: dataframe with columns:  'timestep', 'region_id', 'region_name', 'region_population', 'S', 'E', I','R', 'H', 'V', 'E_per_100k'
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
    """ transforms a data frame to 3D array
    
    Parameters
        df:  dataframe 
        coloumn_names: string indicating the column names of the data that will be transformed e.g 'SEIRHQ'. 
    Returns
         3D array with shape (number of time steps, number of compartments, number of regions)
    """
    l = []
    df = df[list(column_names)]
    for i in range(0, len(df), 356):
        l.append(np.transpose(df.iloc[i:i+356].to_numpy()))
    return np.array(l)

def seir_plot_one_cell(history, cellid):
    """ plots SEIR curves for a single region

    Parameters
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        cellid: index of the region to plot SEIR curves
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
    """ plots accumulated SEIR curves
    
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

