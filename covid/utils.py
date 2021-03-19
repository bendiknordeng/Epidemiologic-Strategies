import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import ast
from collections import namedtuple

def read_config(filepath="configs/baseline.txt"):
    file = open(filepath, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return namedtuple('Config', dictionary.keys())(**dictionary)
    # dicts_from_file now contains the dictionaries created from the text file

def generate_od_matrix(num_time_steps, num_regions):
    """ generate an OD-matrix used for illustrative purposes only
    Paramters:
        num_regions:
        num_time_steps:
    Returns:
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

def generate_vaccine_matrix(num_time_steps, num_regions):
    """ generate an vaccine matrix used for illustrative purposes only
    Paramters:
        num_counties:
        num_time_steps:
    Returns:
        An vaccine matrix with dimensions (num_time_steps, num_counties) 
    """
    a = [[10 for x in range(num_regions)] for i in range (num_time_steps)]
    return np.array(a)
    
def write_pickle(filepath, arr):
    """ writes arr as a pickle to filepath
    Paramters:
        filepath: string filepath
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ read pickle from filepath
    Paramters:
        filepath: string filepath
    Returns:
        arr: array that is read from filepath 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)

def transform_res_to__df(res, column_names, region_names, region_population):
    """ Transform a 3D maattrix that is the result from SEIR modelling to a pandas dataframe
    Paramters:
        res: 3D matrix e.g (250, 6, 11)
        coloumn_names: string that represents the column names (e.g 'SEIRHQ').  Should have the same length as res[1]
        region_names: list with strings that represents all the regions. Should have the same length as res[2]
    Returns:
        df: dataframe that represents the 3D matrix (2750 (res[0]*res[2]), 6 (res[1]))
    """
    A = res.transpose(0,2,1)
    (a,b,c) = A.shape 
    B = A.reshape(-1,c) 
    df = pd.DataFrame(B, columns=list(column_names))
    df['Region'] = np.tile(np.array(region_names), a)
    df['Population'] = np.tile(np.array(region_population), a)
    df['Day'] = np.floor_divide(df.index.values, b)
    return df 

def seir_plot_one_cell(res, cellid):
        """ Plots SIR for a single cell
        self.parameters:
        res: [3D array, comself.partment_id]
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
    """ Plots the epidemiological curves
    self.parameters:
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
