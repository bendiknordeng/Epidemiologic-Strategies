import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import ast
from collections import namedtuple
import os


def create_named_tuple(filepath):
    """ Generate a namedtuple from a txt file

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
    """ Generate an OD-matrix used for illustrative purposes only

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
    """ Generate an OD-matrix used for illustrative purposes only

    Parameters
        num_time_steps: int indicating number of time periods e.g 28
        population: a dataframe with region_id, region_name and population
        fpath_muncipalities_commute: filepath to commuting data between regions
    Returns
        An OD-matrix with dimensions (num_time_steps, num_regions, num_regions) indicating travel in percentage of current population 
    """

    df = pd.read_csv(fpath_muncipalities_commute, usecols=[1,2,3])
    df['from'] = df['from'].str.lstrip("municip municip0").astype(int)
    df['to'] = df['to'].str.lstrip("municip municip0").astype(int)
    decision_period = 28
    region_id = population.region_id.to_numpy()
    od = np.zeros((decision_period, len(region_id), len(region_id)))
    morning = np.zeros((len(region_id), len(region_id)))
    
    for id in region_id:
        i = np.where(region_id == id)
        filtered = df.where(df["from"] == id).dropna()
        to, n = filtered['to'].to_numpy(int), filtered['n'].to_numpy()
        for k in range(len(to)):
            j = np.where(region_id == to[k])
            morning[i,j] = n[k]

    for i in range(len(region_id)):
        sum_travel = morning[i,:].sum()
        pop_i = population.iloc[i].population
        morning[i,i] = pop_i - sum_travel
    
    afternoon = np.copy(morning.T)
    
    for i in range(len(region_id)):
        morning[i,:] = morning[i,:]/morning[i,:].sum()
        afternoon[i,:] = afternoon[i,:]/afternoon[i,:].sum() 

    midday = np.zeros((len(region_id), len(region_id)))
    np.fill_diagonal(midday, np.ones(len(region_id)))  # ([morning[:,i].sum() for i in range(len(region_id))])) 
    night = np.copy(midday)

    # fill od matrices with correct matrix
    for i in range(num_time_steps):
        if i >= 20: # weekend: no travel
            od[i] = night
        elif (i)%4 == 0: # 0000-0600
            od[i] = night
        elif (i-1)%4 == 0: # 0600-1200
            od[i] = morning
        elif (i-2)%4 == 0: # 1200-1800
            od[i] = midday
        elif (i-3)%4 == 0: # 1800-0000
            od[i] = afternoon
    
    return od

def write_pickle(filepath, arr):
    """ Writes an array to file as a pickle

    Parameters
        filepath: string file path
        arr: array that is written to file 
    """
    with open(filepath,'wb') as f:
        pkl.dump(arr, f)

def read_pickle(filepath):
    """ Read pickle and returns an array

    Parameters
        filepath: string file path
    Returns
        arr: array that is read from file path 
    """
    with open(filepath,'rb') as f:
        return pkl.load(f)

def transform_history_to_df(time_step, history, population, column_names):
    """ Transforms a 3D array that is the result from SEIR modelling to a pandas dataframe

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
    """ Transforms a dataframe to 3D array
    
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


def transform_historical_df_to_history(df):
    """ Transforms a dataframe to 3D array
    
    Parameters
        df:  dataframe of real historical covid data for Norway's municipalities
    Returns
         3D array with shape (number of time steps, number of compartments, number of regions)
    """
    # Add leading zero for municipality id
    df['kommune_no'] = df['kommune_no'].apply(lambda x: '{0:0>4}'.format(x)) 
    df = df[['kommune_no', 'cases']]
    df = df.rename(columns={'cases': 'I'})
    return transform_df_to_history(df, 'I')
    
def create_population(fpath_muncipalities_names, fpath_muncipalities_pop):
    """ Load population information into dataframes

    Parameters
        fpath_muncipalities_names: file path to region names
        fpath_muncipalities_pop: file path to population data
    Returns
        population: a dataframe with region_id, region_name and population
    """
    region = pd.read_csv(fpath_muncipalities_names, delimiter=",").drop_duplicates()
    region.rename(columns={"Kommunenr. 2020": "region_id", "Kommunenavn 2020": "region"}, inplace=True)
    region.region_id = region.region_id.astype(int)

    # create population 
    region_population = pd.read_csv(fpath_muncipalities_pop, delimiter=";", skiprows=1)
    region_ids = []
    for r in region_population.region.str.split(" "):
        region_ids.append(int(r[0]))
    region_population["region_id"] = region_ids
    region_population = region_population[["region_id", "Befolkning per 1.1. (personer) 2020"]].rename(columns={ "Befolkning per 1.1. (personer) 2020": "population"})
    population_df = pd.merge(region_population, region, on='region_id', sort=True)
    return population_df

def write_history(write_weekly, history, population, time_step, path_results_weekly, path_results_history):
    """ Write history of simulation to csv files

    Parameters
        write_weekly: bool value specifying if time resolution on csv file should be weekly
        history: the complete 3d array of simulation, in every time step
        population: a dataframe with region_id, region_name and population
        time_step: the current time step of the simulation
        path_results_weekly: path to store weekly results of simulation
        path_results_history: path to store complete results of simulation
    """

    if write_weekly:
        latest_df = transform_history_to_df(time_step, np.expand_dims(history[-1], axis=0), population, 'SEAIQRDVH')
        if os.path.exists(path_results_weekly):
            if time_step == 0: # block to remove old csv file if new run is executed 
                os.remove(path_results_weekly)
                latest_df.to_csv(path_results_weekly)
            else:
                latest_df.to_csv(path_results_weekly, mode='a', header=False)
        else:
            latest_df.to_csv(path_results_weekly)
    else:
        history_df = transform_history_to_df(time_step, history, population, 'SEAIQRDVH')
        if os.path.exists(path_results_history):
            if time_step == 0: # block to remove old csv file if new run is executed
                os.remove(path_results_history)
                history_df.to_csv(path_results_history)
            else:
                history_df.to_csv(path_results_history, mode='a', header=False)
        else:
            history_df.to_csv(path_results_history)
