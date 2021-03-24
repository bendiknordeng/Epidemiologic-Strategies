import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import ast
from collections import namedtuple

def read_paths(path):
    """ generate a namedtuple from a txt file
    Parameters
        path: file path to .txt file
    Returns
        A namedtuple representing each path needed for system execution
    """
    file = open(path, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return namedtuple('Config', dictionary.keys())(**dictionary)
    # dicts_from_file now contains the dictionaries created from the text file

def read_config(fpath_config):
    """ generate a namedtuple from a config file
    Parameters
        fpath_config: file path to .txt config file
    Returns
        A namedtuple representing config file
    """
    file = open(fpath_config, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return namedtuple('Config', dictionary.keys())(**dictionary)
    # dicts_from_file now contains the dictionaries created from the text file

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

def generate_vaccine_matrix(num_time_steps):
    """ generate an vaccine matrix used for illustrative purposes only
    Parameters
        num_regions: number of regions e.g 356
        num_time_steps: number of time periods e.g 28
    Returns
        An vaccine matrix with dimensions (num_time_steps, num_counties) 
    """
    a = [40 for i in range (num_time_steps)]
    return np.array(a)
    
def write_pickle(filepath, arr):
    """ saves an array as a pickle to filepath
    Parameters
        filepath: string filepath
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ read pickle from filepath
    Parameters
        filepath: string filepath
    Returns
        arr: array that is read from filepath 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)

def transform_history_to_df(time_step, history, column_names):
    """ transforms a 3D matrix that is the result from SEIR modelling to a pandas dataframe
    Parameters
        res: 3D matrix e.g (250, 6, 11)
        coloumn_names: string that represents the column names (e.g 'SEIRHQ').  Should have the same length as res[1]
        region_names: list with strings that represents all the regions. Should have the same length as res[2]
    Returns
        df: dataframe that represents the 3D matrix e.g (2750 (res[0]*res[2]), 6 (res[1]))
    """
    A = history.transpose(0,2,1)
    (a,b,c) = A.shape 
    B = A.reshape(-1,c) 
    df = pd.DataFrame(B, columns=list(column_names))
    region_numbers = [x for x in range(356)]
    df['Region'] = np.tile(np.array(region_numbers), a)
    df['Timestep'] = np.floor_divide(df.index.values, b) + time_step
    return df

def seir_plot_one_cell(res, cellid):
    """ plots SEIR curves for a single region
    Parameters
        res: [3D array, comself.partment_id]
        cellid: index to the region to plot in the res 3d-array
    """
    plt.plot(res[::12, 0, cellid], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1, cellid], color='g', label='E')
    plt.plot(res[::12, 2, cellid], color='b', label='I')
    plt.plot(res[::12, 3, cellid], color='y', label='R')
    plt.plot(res[::12, 4, cellid], color='c', label='H')
    plt.plot(res[::12, 5, cellid], color='m', label='V')
    plt.legend()
    plt.show()

def seir_plot(res):
    """ plots SEIR curves for all regions 
    Parameters
        res: [3D array, comself.partment_id]
    """
    plt.plot(res[::12, 0], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.plot(res[::12, 5], color='m', label='V')
    plt.legend()
    plt.show()