import pandas as pd
import numpy as np
import pickle as pkl
import ast
from collections import namedtuple
import os
from datetime import datetime, timedelta


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
    afternoon = np.copy(morning.T)
    
    morning = np.transpose(morning.T / population.population.to_numpy())
    afternoon = np.transpose(afternoon.T / population.population.to_numpy())

    midday = np.zeros((len(region_id), len(region_id)))
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

def transform_path_to_numpy(path):
    history = []
    new_infections = []
    for state in path:
        history.append(state.get_compartments_values())
        new_infections.append(state.new_infected)
    return np.array(history), np.array(new_infections)

def transform_history_to_df(time_step, history, population, column_names):
    """ transforms a 3D array that is the result from SEIR modelling to a pandas dataframe

    Parameters
        time_step: integer used to indicate the current time step in the simulation
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        population: DataFrame with region_id, region_names, age_group_population and total population (quantity)
        coloumn_names: string that represents the column names e.g 'SEIRHQ'. 
    Returns
        df: dataframe with columns:  'timestep', 'region_id', 'region_name', 'region_population', 'S', 'E', I','R', 'V', 'E_per_100k'
    """
    A = history.transpose(0,2,3,1)
    (periods, regions, age_groups, compartments) = A.shape 
    B = A.reshape(-1, compartments)
    df = pd.DataFrame(B, columns=list(column_names))
    df['date'] = [get_date("20200221", int(time_step)) for time_step in np.floor_divide(df.index.values, regions*age_groups) + time_step//4]
    df['time_step'] = np.floor_divide(df.index.values, regions*age_groups)*4 + time_step
    df['age_group'] = np.tile(np.array(population.columns[2:-1]), periods*regions)
    df['region_id'] = np.array([[[r]*age_groups for r in population.region_id] for _ in range(periods)]).reshape(-1)
    df['region_name'] = np.array([[[r]*age_groups for r in population.region] for _ in range(periods)]).reshape(-1)
    df['region_population'] = np.array([[[r]*age_groups for r in population.population] for _ in range(periods)]).reshape(-1)
    df['E1_per_100k'] = 1e5*df.E1/df.region_population
    return df[['date', 'time_step', 'region_id', 'region_name', 'age_group', 'region_population'] + list(column_names) + ['E1_per_100k']]

def transform_df_to_history(df, column_names, n_regions, n_age_groups):
    """ transforms a dataframe to 3D array
    
    Parameters
        df:  dataframe 
        coloumn_names: string indicating the column names of the data that will be transformed e.g 'SEIRHQ'. 
    Returns
         3D array with shape (number of time steps, number of compartments, number of regions)
    """

    df = df[list(column_names)]
    l = []
    for i in range(0, len(df), n_age_groups):
        l.append(df.iloc[i:i+5,:].sum())
    compressed_df = pd.DataFrame(l)

    l = []
    for i in range(0, len(compressed_df), n_regions):
        l.append(np.transpose(compressed_df.iloc[i:i+356].to_numpy()))

    return np.array(l)


def transform_historical_df_to_history(df):
    """ transforms a dataframe to 3D array
    
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

def generate_custom_population(bins, labels, path_pop, path_region_names):
    """ generates age divided population

    Parameters
        bins: 1D array of bins in which to divide population
        labels: list of strings, names of age groups
        path_pop: path to population data
        path_region_names: path to region name data
    Returns
        dataframe with age divided population
    """
    total_pop = pd.read_csv(path_pop)
    age_divided = pd.DataFrame(total_pop.groupby(['region_id', pd.cut(total_pop["age"], bins=bins+[110], labels=labels, include_lowest=True)]).sum('population')['population'])
    age_divided.reset_index(inplace=True)
    age_divided = age_divided.pivot(index='region_id', columns=['age'])['population']
    region_names_id = pd.read_csv(path_region_names, delimiter=",").drop_duplicates()
    df = pd.merge(region_names_id, age_divided, on="region_id", how='right', sort=True)
    df['population'] = df.loc[:,df.columns[2:2+len(labels)]].sum(axis=1)
    return df

def generate_labels_from_bins(bins):
    """ generates labels for population dataframe

    Parameters
        bins: 1D array of bins to divide population
    Returns
        labels defining the population bins
    """
    labels = []
    for i in range(len(bins)-1):
        if i == 0:
            labels.append(str(bins[i])+"-"+str(bins[i+1]))
        else:
            labels.append(str(bins[i]+1)+"-"+str(bins[i+1]))
    labels.append(str(bins[-1]+1)+"+")
    return labels

def generate_contact_matrices(bins, labels, population, country=None):
    df = pd.read_csv('data/contact_data.csv')
    if country: df = df[df.country == country]
    df.contact_age_0 = pd.cut(df['contact_age_0'], bins=bins+[110], labels=labels, include_lowest=True)
    df.contact_age_1 = pd.cut(df['contact_age_1'], bins=bins+[110], labels=labels, include_lowest=True)
    df_mat = pd.DataFrame(df[df.columns[:-2]].groupby(['contact_age_0', 'contact_age_1']).mean()).reset_index()
    N = population[population.columns[2:-1]].sum().to_numpy()
    matrices = []
    for col in ['home', 'school', 'work', 'transport', 'leisure']:
        matrix = pd.pivot_table(df_mat, values=col, index='contact_age_0', columns='contact_age_1').to_numpy()
        symmetric_matrix = np.zeros((matrix.shape))
        for i in range(len(N)):
            for j in range(len(N)):
                symmetric_matrix[i][j] = 1/(N[i]+N[j]) * (matrix[i][j] * N[i] + matrix[j][i] * N[j])
        matrices.append(symmetric_matrix)
    return matrices

def get_age_group_flow_scaling(bins, labels, population):
    percent_commuters = 0.36 # numbers from SSB
    df = pd.read_csv('data/employed_per_age.csv')
    df.age = pd.cut(df['age'], bins=bins+[110], labels=labels, include_lowest=True)
    commuters = df.groupby('age').sum()['employed'].to_numpy() * percent_commuters
    sum_age_groups = population[population.columns[2:-1]].sum().to_numpy()
    age_group_commuter_percent = commuters/sum_age_groups
    return age_group_commuter_percent/age_group_commuter_percent.sum()

def get_age_group_fatality_prob(bins, labels):
    df = pd.read_csv('data/death_by_age.csv')
    df.age = pd.cut(df['age'], bins=bins+[110], labels=labels, include_lowest=True)
    infected = df.groupby('age').sum()['infected'].to_numpy()
    dead = df.groupby('age').sum()['dead'].to_numpy()
    return dead/infected

def get_historic_data(path):
    historic_data = pd.read_csv(path)  # set to None if not used
    historic_data.date = pd.to_datetime(historic_data.date)
    return historic_data

def write_history(write_weekly, history, population, time_step, results_weekly, results_history, labels):
    """ write history array to csv

    Parameters
        write_weekly: Bool, if the results should be written on weekly basis
        history: 3D array with shape (number of time steps, number of compartments, number of regions)
        population: pd.DataFrame with columns region_id, region_name, population (quantity)
        time_step: Int, indicating the time step of the simulation
        results_weekly: Bool, indicating if weekly results exists. Stored data is removed if True.
        results_history: Bool, indicating if results exists. Stored data is removed if True.
        compartments:
    """
    if write_weekly:
        weekly_new_infected = history[:,-1].sum(axis=0)
        last_day = np.expand_dims(history[-1], axis=0)
        last_day[:,-1] = weekly_new_infected
        latest_df = transform_history_to_df(time_step, last_day, population, labels)
        if os.path.exists(results_weekly):
            if time_step == 0: # block to remove old csv file if new run is executed 
                os.remove(results_weekly)
                latest_df.to_csv(results_weekly, index=False)
            else:
                latest_df.to_csv(results_weekly, mode='a', header=False, index=False)
        else:
            latest_df.to_csv(results_weekly, index=False)
    else:
        daily_df = transform_history_to_df(time_step, history, population, labels)
        if os.path.exists(results_history):
            if time_step == 0: # block to remove old csv file if new run is executed
                os.remove(results_history)
                daily_df.to_csv(results_history, index=False)
            else:
                daily_df.to_csv(results_history, mode='a', header=False, index=False)
        else:
            daily_df.to_csv(results_history, index=False)
    
def generate_weekly_data(fpath_fhi_data_daily, fpath_fhi_data_weekly):
    """ aggregates daily data to weekly data and saves it
    
    Parameters
        fpath_fhi_data_daily: str, filepath to daily historical FHI data. Saved as .xlsx file 
        fpath_fhi_data_weekly: str, filepath where weekly data is written to. Saved as .csv file

    """
    df = pd.read_excel(fpath_fhi_data_daily)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    resample_dict ={'year': 'last', 
                    'week':'last', 
                    'r0_average': 'mean', 
                    'r0_conf_95_low':'mean',
                    'r0_conf_95_high':'mean', 
                    'H_cumulative':'last', 
                    'H_new':'sum', 
                    'ICU_cummulative':'last',
                    'ICU_new':'sum', 
                    'I_cumulative':'last',
                    'I_new':'sum', 
                    'D_cumulative':'last', 
                    'D_new':'sum',
                    'V_1_cumulative':'last', 
                    'V_2_cumulative':'last', 
                    'V_1_new':'sum', 
                    'V_2_new':'sum',
                    'vaccine_supply_new':'sum',
                    'alpha_s':'mean',
                    'alpha_e1':'mean',
                    'alpha_e2':'mean',
                    'alpha_a':'mean',
                    'alpha_i':'mean',
                    'w_c1': 'mean',
                    'w_c2': 'mean',
                    'w_c3': 'mean',
                    'w_c4': 'mean'}

    df2 = df.groupby(['year','week']).agg(resample_dict)
    df2.to_csv(fpath_fhi_data_weekly, index=False)

def get_date(start_date, time_step=0):
    """ gets current date for a simulation time step
    Parameters
        start_date: str indicating start date of simulation in the format 'YYYYMMDD' 
        time_delta: int indicating number of days from simulation start
    Returns
        datdatetime.date object with a given date
    """
    dt = datetime.strptime(start_date, '%Y%m%d').date()
    dt += timedelta(days=time_step)
    return dt

def print_results(history, new_infections, population, age_labels, policy, save_to_file=False):
    total_pop = np.sum(population.population)
    infected = [new_infections[:,:,i].sum(axis=0).sum(axis=0) for i in range(len(age_labels))]
    vaccinated = history[-1,7,:,:].sum(axis=0)
    dead = history[-1,6,:,:].sum(axis=0)
    age_total = population[age_labels].sum().to_numpy()
    columns = ["Age group", "Infected", "Vaccinated", "Dead", "Total"]
    result = f"{columns[0]:<9} {columns[1]:>20} {columns[2]:>20} {columns[3]:>20}\n"
    for i in range(len(age_labels)):
        age_pop = np.sum(population[age_labels[i]])
        result += f"{age_labels[i]:<9}"
        result += f"{infected[i]:>12,.0f} ({100 * infected[i]/age_pop:>5.2f}%)"
        result += f"{vaccinated[i]:>12,.0f} ({100 * vaccinated[i]/age_pop:>5.2f}%)"
        result += f"{dead[i]:>12,.0f} ({100 * dead[i]/age_pop:>5.2f}%)\n"
    result += f"{'All':<9}"
    result += f"{np.sum(infected):>12,.0f} ({100 * np.sum(infected)/total_pop:>5.2f}%)"
    result += f"{np.sum(vaccinated):>12,.0f} ({100 * np.sum(vaccinated)/total_pop:>5.2f}%)"
    result += f"{np.sum(dead):>12,.0f} ({100 * np.sum(dead)/total_pop:>5.2f}%)"
    print(result)

    if save_to_file:
        data = np.array([age_labels, np.round(infected), np.round(vaccinated), np.round(dead), age_total]).T
        df = pd.DataFrame(columns=columns, data=data)
        for col in columns[1:]:
            df[col] = df[col].astype(float)
            df[col] = df[col].astype(int)
        total = df[df.columns[1:]].sum()
        total["Age group"] = "All"
        df = df.append(total, ignore_index=True)
        df.to_csv(f"results/final_results_{policy}.csv", index=False)


def create_timeline(horizon, decision_period):
    nr_waves = int(np.random.poisson(horizon/13)) # assumption: a wave happens on average every 13 weeks
    duration_waves = [int(np.random.exponential(2*decision_period)) for _ in range(nr_waves)] # assumption, a wave lasts on average 2 weeks
    mean = (horizon*decision_period - np.sum(duration_waves))/(nr_waves+1) # evenly distributed time between waves
    std_dev = mean/3
    time_between_waves = [int(np.random.normal(mean, std_dev)) for _ in range(nr_waves+1)] # time between waves, normally distributed
    timeline = [[time_between_waves[0], 0]]
    for i in range(len(duration_waves)):
        timeline.append([duration_waves[i]/2 + timeline[-1][0], 1]) # wave incline
        timeline.append([duration_waves[i]/2 + timeline[-1][0], -1]) # wave decline
        timeline.append([time_between_waves[i+1] + timeline[-1][0], 0]) # neutral start
    timeline.append([horizon*decision_period, 0])
    return np.array(timeline, int)
