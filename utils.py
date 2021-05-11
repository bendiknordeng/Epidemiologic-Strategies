import pandas as pd
import numpy as np
import pickle as pkl
import ast
from collections import namedtuple
import os
from datetime import datetime, timedelta
from scipy import stats as sps
import plot
from scipy.stats import skewnorm
import json
from collections import Counter
import geopandas as gpd


class tcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_named_tuple(name, filepath):
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
    return namedtuple(name, dictionary.keys())(**dictionary)

paths = create_named_tuple('paths', 'filepaths.txt')

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

def generate_commuter_matrix(age_flow_scaling):
    """ generate an OD-matrix used for illustrative purposes only

    Parameters
        num_time_steps: int indicating number of time periods e.g 28
        population: a dataframe with region_id, region_name and population
    Returns
        An OD-matrix with dimensions (num_time_steps, num_regions, num_regions) indicating travel in percentage of current population 
    """
    df = pd.read_csv(paths.municipalities_commuters)
    commuters = df.pivot(columns='to', index='from', values='n').fillna(0).values
    visitors = np.array([commuters.sum(axis=0) * age_flow_scaling[i] for i in range(len(age_flow_scaling))]).T
    visitors[np.where(visitors == 0)] = 1
    return visitors, commuters

def write_pickle(filepath, object):
    """ writes an array to file as a pickle

    Parameters
        filepath: string file path
        arr: array that is written to file 
    """
    with open(filepath, 'wb') as f:
        pkl.dump(object, f)

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
        history: numpy.ndarray with shape (number of time steps, number of compartments, number of regions)
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
    """ transforms a dataframe to numpy.ndarray
    
    Parameters
        df:  dataframe 
        column_names: string indicating the column names of the data that will be transformed e.g 'SEIRHQ'. 
    Returns
         numpy.ndarray with shape (number of time steps, number of compartments, number of regions)
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
    """ transforms a dataframe to numpy.ndarray
    
    Parameters
        df:  dataframe of real historical covid data for Norway's municipalities
    Returns
         numpy.ndarray with shape (number of time steps, number of compartments, number of regions)
    """
    # Add leading zero for municipality id
    df['kommune_no'] = df['kommune_no'].apply(lambda x: '{0:0>4}'.format(x)) 
    df = df[['kommune_no', 'cases']]
    df = df.rename(columns={'cases': 'I'})
    return transform_df_to_history(df, 'I')

def generate_custom_population(bins, labels):
    """ generates age divided population

    Parameters
        bins: numpy.ndarray of bins in which to divide population
        labels: list of strings, names of age groups
    Returns
        dataframe with age divided population
    """
    total_pop = pd.read_csv(paths.age_divided_population)
    age_divided = pd.DataFrame(total_pop.groupby(['region_id', pd.cut(total_pop["age"], bins=bins+[110], labels=labels, include_lowest=True)]).sum('population')['population'])
    age_divided.reset_index(inplace=True)
    age_divided = age_divided.pivot(index='region_id', columns=['age'])['population']
    region_names_id = pd.read_csv(paths.municipalities_names, delimiter=",").drop_duplicates()
    df = pd.merge(region_names_id, age_divided, on="region_id", how='right', sort=True)
    df['population'] = df.loc[:,df.columns[2:2+len(labels)]].sum(axis=1)
    return df

def generate_labels_from_bins(bins):
    """ generates labels for population dataframe

    Parameters
        bins: numpy.ndarray of bins to divide population
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
    df = pd.read_csv(paths.contact_data)
    if country: df = df[df.country == country]
    df.contact_age_0 = pd.cut(df['contact_age_0'], bins=bins+[110], labels=labels, include_lowest=True)
    df.contact_age_1 = pd.cut(df['contact_age_1'], bins=bins+[110], labels=labels, include_lowest=True)
    df_mat = pd.DataFrame(df[df.columns[:-2]].groupby(['contact_age_0', 'contact_age_1']).sum()).reset_index()
    N_survey_0 = df.contact_age_0.value_counts()[labels]
    pop_0 = [N_survey_0[l[1].contact_age_0] for l in df_mat.iterrows()]
    df_mat['pop_0'] = pop_0
    for col in df_mat.columns[2:-1]:
        df_mat[col] = df_mat[col]/(df_mat.pop_0)

    N_eu = pd.read_csv(paths.europe_data)
    N_eu.age = pd.cut(N_eu['age'], bins=bins+[110], labels=labels, include_lowest=True)
    N_eu = N_eu.groupby('age').sum()['population']
    N_eu_tot = N_eu.sum()
    N_norway = population[population.columns[2:-1]].sum()
    N_norway_tot = np.sum(N_norway)

    matrices = []
    for col in ['home', 'school', 'work', 'public']:
        matrix = pd.pivot_table(df_mat, values=col, index='contact_age_0', columns='contact_age_1')
        corrected_matrix = np.zeros((matrix.shape))
        for i, a_i in enumerate(labels):
            for j, a_j in enumerate(labels):
                corrected_matrix[i][j] = matrix[a_i][a_j] * (N_eu_tot * N_norway[a_j])/(N_eu[a_j] * N_norway_tot) # Density correction
        symmetric_matrix = np.zeros((matrix.shape))
        for i, a_i in enumerate(labels):
            for j, a_j in enumerate(labels):
                symmetric_matrix[i][j] = 1/(N_norway[a_i]+N_norway[a_j]) * (corrected_matrix[i][j] * N_norway[a_i] + corrected_matrix[j][i] * N_norway[a_j]) # Symmetry
        matrices.append(symmetric_matrix)
    return matrices

def generate_weighted_contact_matrix(C, contact_weights):
        """ Scales the contact matrices with weights, and return the weighted contact matrix used in modelling

        Parameters
            weights: list of floats indicating the weight of each contact matrix for school, workplace, etc. 
        Returns
            weighted contact matrix used in modelling
        """
        return np.sum(np.array([np.array(C[i])*contact_weights[i] for i in range(len(C))]), axis=0)

def get_age_group_flow_scaling(bins, labels, population):
    percent_commuters = 0.36 # numbers from SSB
    df = pd.read_csv(paths.employed_by_age)
    df.age = pd.cut(df['age'], bins=bins+[110], labels=labels, include_lowest=True)
    commuters = df.groupby('age').sum()['employed'].to_numpy() * percent_commuters
    sum_age_groups = population[population.columns[2:-1]].sum().to_numpy()
    age_group_commuter_percent = commuters/sum_age_groups
    return age_group_commuter_percent/age_group_commuter_percent.sum()

def get_age_group_fatality_prob(bins, labels):
    df = pd.read_csv(paths.deaths_by_age)
    df.age = pd.cut(df['age'], bins=bins+[110], labels=labels, include_lowest=True)
    infected = df.groupby('age').sum()['cases'].to_numpy()
    dead = df.groupby('age').sum()['deaths'].to_numpy()
    return dead/infected * (1.9/3.6) # Our world data on hospital beds per 1000 (California/Norway)

def get_historic_data():
    historic_data = pd.read_csv(paths.fhi_data_daily)  # set to None if not used
    historic_data.date = pd.to_datetime(historic_data.date)
    return historic_data

def write_history(write_weekly, history, population, time_step, results_weekly, results_history, labels):
    """ write history array to csv

    Parameters
        write_weekly: Bool, if the results should be written on weekly basis
        history: numpy.ndarray with shape (number of time steps, number of compartments, number of regions)
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

def transform_path_to_numpy(path):
    history = []
    new_infections = []
    for state in path:
        history.append(state.get_compartments_values())
        new_infections.append(state.new_infected)
    return np.array(history), np.array(new_infections)

def print_results(state, population, age_labels, policy, save_to_file=False):
    total_pop = np.sum(population.population)
    infected = state.total_infected.sum(axis=0)
    vaccinated = state.V.sum(axis=0)
    dead = state.D.sum(axis=0)
    age_total = population[age_labels].sum().to_numpy()
    columns = ["Age group", "Infected", "Vaccinated", "Dead", "Total"]
    result = f"\n\n{columns[0]:<9} {columns[1]:>20} {columns[2]:>20} {columns[3]:>20}\n"
    for i in range(len(age_labels)):
        age_pop = np.sum(population[age_labels[i]])
        result += f"{age_labels[i]:<9}"
        result += f"{infected[i]:>12,.0f} ({100 * infected[i]/age_pop:>5.2f}%)"
        result += f"{vaccinated[i]:>12,.0f} ({100 * vaccinated[i]/age_pop:>5.2f}%)"
        result += f"{dead[i]:>12,.0f} ({100 * dead[i]/age_pop:>5.2f}%)\n"
    result += f"{'All':<9}"
    result += f"{np.sum(infected):>12,.0f} ({100 * np.sum(infected)/total_pop:>5.2f}%)"
    result += f"{np.sum(vaccinated):>12,.0f} ({100 * np.sum(vaccinated)/total_pop:>5.2f}%)"
    result += f"{np.sum(dead):>12,.0f} ({100 * np.sum(dead)/total_pop:>5.2f}%)\n"
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

def get_average_results(final_states, population, age_labels, policy, save_to_file=False):
    final_infected = []
    final_vaccinated = []
    final_dead = []
    for state in final_states:
        final_infected.append(state.total_infected.sum(axis=0))
        final_vaccinated.append(state.V.sum(axis=0))
        final_dead.append(state.D.sum(axis=0))
    average_infected = np.average(np.array(final_infected), axis=0)
    average_vaccinated = np.average(np.array(final_vaccinated), axis=0)
    average_dead = np.average(np.array(final_dead), axis=0)

    std_infected = np.std(np.array(final_infected), axis=0)
    std_vaccinated = np.std(np.array(final_vaccinated), axis=0)
    std_dead = np.std(np.array(final_dead), axis=0)

    total_pop = np.sum(population.population)
    age_total = population[age_labels].sum().to_numpy()
    
    total_std_infected = np.sqrt(np.sum(np.square(std_infected) * age_total)/ total_pop)
    total_std_vaccinated = np.sqrt(np.sum(np.square(std_vaccinated) * age_total)/ total_pop)
    total_std_dead = np.sqrt(np.sum(np.square(std_dead) * age_total)/ total_pop)

    columns = ["Age group", "Infected", "Vaccinated", "Dead", "Total"]
    result = f"\n{columns[0]:<10} {columns[1]:^36} {columns[2]:^36} {columns[3]:^36}\n"
    for i in range(len(age_labels)):
        age_pop = np.sum(population[age_labels[i]])
        result += f"{age_labels[i]:<10}"
        result += f"{average_infected[i]:>12,.0f} ({100 * average_infected[i]/age_pop:>5.2f}%) SD: {std_infected[i]:>9.2f}"
        result += f"{average_vaccinated[i]:>12,.0f} ({100 * average_vaccinated[i]/age_pop:>5.2f}%) SD: {std_vaccinated[i]:>9.2f}"
        result += f"{average_dead[i]:>12,.0f} ({100 * average_dead[i]/age_pop:>5.2f}%) SD: {std_dead[i]:>9.2f}\n"
    result += f"{'All':<10}"
    result += f"{np.sum(average_infected):>12,.0f} ({100 * np.sum(average_infected)/total_pop:>5.2f}%) SD: {total_std_infected:>9.2f}"
    result += f"{np.sum(average_vaccinated):>12,.0f} ({100 * np.sum(average_vaccinated)/total_pop:>5.2f}%) SD: {total_std_vaccinated:>9.2f}"
    result += f"{np.sum(average_dead):>12,.0f} ({100 * np.sum(average_dead)/total_pop:>5.2f}%) SD: {total_std_dead:>9.2f}"
    print(result)
    
    if save_to_file:
        data = np.array([age_labels, np.round(average_infected), np.round(average_vaccinated), np.round(average_dead), age_total]).T
        df = pd.DataFrame(columns=columns, data=data)
        for col in columns[1:]:
            df[col] = df[col].astype(float)
            df[col] = df[col].astype(int)
        total = df[df.columns[1:]].sum()
        total["Age group"] = "All"
        df = df.append(total, ignore_index=True)
        df.to_csv(f"results/final_results_{policy}.csv", index=False)

def get_wave_timeline(horizon, decision_period, periods_per_day, *args):
    """generates a wave timeline and a wave state timeline over the simulation horizon

    Args:
        horizon (int): simulation horizon (weeks)
        decision_period (int): number of periods within a week e.g 28
        periods_per_day (int): number of periods per day e.g 4 

    Returns:
        wave_timeline (list(float)): r effective values for each week over the simulation horizon
        wave_state_timeline (list(str)): characters indicating the wave state for each week of the simulation horizon

    """
    with open('data/waves/wave_parameters.json') as file:
        data = json.load(file)
    transition_mat = pd.read_csv(paths.wave_transition, index_col=0).T.to_dict()
    decision_period_days = int(decision_period/periods_per_day)
    wave_timeline = np.zeros(horizon)
    current_state = 'U'
    wave_state_count = [current_state]
    wave_state_timeline = []
    len_current_state = 0
    from_start = True
    i = 0
    if args:
        i = args[2]
        wave_timeline[:i] = args[0][:i]
        wave_state_timeline = args[1][:i]
        current_state = wave_state_timeline[-1]
        previous_states = list(set(wave_state_timeline)-{current_state})
        len_current_state = min([wave_state_timeline[::-1].index(state) for state in previous_states]) // decision_period_days
        wave_state_count = [wave_state_timeline[0]]
        for ws in wave_state_timeline[1:]:
            if ws != wave_state_count[-1]:
                wave_state_count.append(ws)
        from_start = False

    while True:
        n_wave = Counter(wave_state_count)[current_state]-1
        params = data['duration'][current_state][str(1 + (n_wave%4))]
        duration = skewnorm.rvs(params['skew'], loc=params['mean'], scale=params['std'])
        duration = min(max(duration, params['min']), params['max']) // decision_period_days
        if not from_start: duration -= len_current_state
        try:
            for week in range(i, i+int(duration)):
                params = data['R'][current_state][str(1 + (n_wave%4))]
                factor = skewnorm.rvs(params['skew'], loc=params['mean'], scale=params['std'])
                factor = min(max(factor, params['min']), params['max'])
                wave_timeline[week] = factor
                wave_state_timeline.append(current_state)
            i += int(duration)
            current_state = np.random.choice(['U', 'D', 'N'], p=list(transition_mat[current_state].values()))
        except:
            break
        wave_state_count.append(current_state)
    return wave_timeline, wave_state_timeline

def get_historic_wave_timeline(horizon):
    df = pd.read_csv(paths.world_r_eff,
        usecols=['country','date','R'],
        squeeze=True
        ).sort_index()
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df_norway = df[df.country == 'Norway']
    d0 = df_norway.date.iloc[0]
    dates = [d0 + pd.Timedelta(i, "W") for i in range(horizon)]
    df_norway_weekly = df_norway[df_norway.date.isin(dates)]
    return df_norway_weekly.R.values

def get_posteriors(new_infected, gamma, r_t_range, sigma=0.15):
    """ function to calculate posteriors

    Parameters
        new_infected: pandas.core.series.Series with new infected per day with date and new_infected as columns
        gamma: 1/recovery period e.g 1/7
        r_t_range: np.array with range R_t can be in
        sigma: Gaussian noise to the prior distribution. represent standard deviation of Gaussian distribution.
    Returns
        posteriors: 
        log_likelihood:

    """
    # (1) Calculate Lambda
    lam = new_infected[:-1].values * np.exp(gamma * (r_t_range[:, None] - 1))
    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(new_infected[1:].values, lam),
        index = r_t_range,
        columns = new_infected.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day. Insert our prior as the first posterior.
    posteriors = pd.DataFrame(index=r_t_range, columns=new_infected.index, data={new_infected.index[0]: prior0})

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(new_infected.index[:-1], new_infected.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator

    return posteriors

def smooth_data(data, window_size=7):
    """ returns smoothed values of a pandas.core.series.Series

    Parameters
        data: pandas.core.series.Series, with date and daily new infected
    Returns
        smoothed: pandas.core.series.Series,  with date and smoothed daily new infected

    """
    smoothed = data.rolling(window_size,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()
    return smoothed

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def highest_density_interval(posteriors, percentile=.9):
    """ finds intervall for the posteriors

    Parameters
        posteriors: pandas.core.frame.DataFrame with posteriors values
        percentile: percentile to find the R_t values for
    Returns
        pandas.core.frame.DataFrame
    """
    if(isinstance(posteriors, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(posteriors[col], percentile=percentile) for col in posteriors], index=posteriors.columns)
    cumsum = np.cumsum(posteriors.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > percentile).nonzero()
    
    # Find the smallest range (highest density)
    try:
        best = (highs - lows).argmin()
        low = posteriors.index[lows[best]]
        high = posteriors.index[highs[best]]
    except:
        low = 0
        high = 0
    
    return pd.Series([low, high], index=[f'Low_{percentile*100:.0f}', f'High_{percentile*100:.0f}'])

def get_r_effective(path, population, config, from_data=False):
    """plots R effective

    Args:
        path ([type]): [description]
        population ([type]): [description]
        config ([type]): [description]
        from_data (bool, optional): Indicating if R effective should be plotted for historical Norwegian data. Defaults to False.
    """
    # Read in data
    if from_data:
        states = pd.read_csv(paths.fhi_data_daily,
                            usecols=['date', 'region', 'I_new'],
                            parse_dates=['date'],
                            index_col=['region', 'date'],
                            squeeze=True).sort_index()
        states = states.astype('float64')
    else:
        regions = np.tile(np.append(population.region_name.to_numpy(), 'NORWAY'), len(path)).T
        I_new = np.array([s.new_infected.sum(axis=1) for s in path])
        total_I_new = I_new.sum(axis=1)
        I_new = np.hstack((I_new,total_I_new.reshape(-1,1)))
        dates = np.array([s.date for s in path]).repeat(I_new.shape[1]).T
        I_new = I_new.reshape(-1).T
        states = pd.DataFrame(data=np.array([regions, dates, I_new]).T, columns=['region','date','I_new'])
        states = states.set_index(['region','date']).sort_values(['region','date'])
        states = states.squeeze().astype('float64')

    # define region and redefine first level in Series
    state_name = 'NORWAY'
    cases = states.xs(state_name).rename(f"{state_name} cases")

    # gets the original and smoothed data 
    smoothed = smooth_data(cases)

    # Need the increment to start with one to calculate posteriors
    idx_start = np.searchsorted(smoothed, 1)
    smoothed = smoothed.iloc[idx_start:]

    # define parameters to calculate posteriors
    R_T_MAX = 10
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
    gamma = 1/(config.presymptomatic_period + config.postsymptomatic_period)
    if from_data: gamma = gamma/config.periods_per_day
    # calculate posteriors 
    posteriors = get_posteriors(smoothed, gamma, r_t_range, sigma=.15)

    # finds the posterior intervals
    hdis = highest_density_interval(posteriors, percentile=0.9)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis], axis=1)

    # plot R_t development
    if len(result > 5):
        plot.plot_rt(result[4:])
    else:
        plot.plot_rt(result)

def get_expected_yll(age_bins, age_labels):
    """ Retrieves the expected years remaining for each age group

    Args:
        age_bins (numpy.ndarray): int describing different age groups
        age_labels (numpy.ndarray): strings with description of different age groups
    Returns:
        int: yll
    """
    df = pd.read_csv(paths.expected_years)
    df.age = pd.cut(df['age'], bins=age_bins+[110], labels=age_labels, include_lowest=True)
    expected_years_remaining = df.groupby('age').mean()['expected_years_remaining'].to_numpy()
    return expected_years_remaining

def calculate_yll(expected_years_remaining, deaths_per_age_group):
    """ Calculates the Years of Life Lost (YLL)

    Args:
        expected_years_remaining (numpy.ndarray): expected years remaining for each age group
        deaths_per_age_group (np.ndarray): accumulated deaths per age_group

    Returns:
        int: total years of life lost
    """
    yll = np.multiply(expected_years_remaining, deaths_per_age_group) 
    return int(np.round(np.sum(yll)))

def load_response_measure_models():
    models = {}
    scalers = {}
    for model_name in ['home', 'school', 'work', 'public', 'alpha', 'movement']:
        models[model_name] = pkl.load(open(f"models/{model_name}_measure_model.sav", 'rb'))
        scalers[model_name] = pkl.load(open(f"models/{model_name}_measure_scaler.sav", 'rb'))
    return models, scalers

def get_avg_std(final_states, population, age_labels):
    final_dead = []
    for state in final_states:
        final_dead.append(state.D.sum(axis=0))
    average_dead = np.average(np.array(final_dead), axis=0)
    std_dead = np.std(np.array(final_dead), axis=0)
    total_pop = np.sum(population.population)
    age_total = population[age_labels].sum().to_numpy()
    total_std_dead = np.sqrt(np.sum(np.square(std_dead) * age_total)/ total_pop)
    return np.sum(average_dead), total_std_dead

def generate_geopandas(pop, fpath_spatial_data):
    pop['region_id'] = pop['region_id'].astype('str')
    pop = pop[['region_id', 'population', 'region_name']]
    gdf = gpd.read_file(fpath_spatial_data)
    gdf = gdf[['region_id', 'geometry']]
    df = pd.DataFrame(gdf)
    gdf = gpd.GeoDataFrame(df.merge(pop, right_on='region_id', left_on='region_id',  suffixes=('', '_y')), geometry='geometry')
    gdf = gdf.dropna()
    gdf = gdf.to_crs(3857)
    return gdf