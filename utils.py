import pandas as pd
import numpy as np
import pickle as pkl
import ast
from collections import namedtuple
import os
from datetime import datetime, timedelta
from pprint import pprint
from scipy.stats import skewnorm
import json
from collections import Counter
import epyestim
from tqdm import tqdm

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
    visitors = np.array([commuters.sum(axis=0) * age_flow_scaling[a] for a in range(len(age_flow_scaling))]).T
    visitors[np.where(visitors == 0)] = np.inf
    age_divided_inflow = np.array([commuters.sum(axis=0) * age_flow_scaling[a] for a in range(len(age_flow_scaling))]).T
    return visitors, commuters, age_divided_inflow

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

def generate_custom_population(bins, labels):
    """ generates age divided population

    Parameters
        bins: numpy.ndarray of bins in which to divide population
        labels: list of strings, names of age groups
    Returns
        dataframe with age divided population
    """
    try: 
        total_pop = pd.read_csv(paths.age_divided_population)
    except:
        total_pop = pd.read_csv("../" + paths.age_divided_population)
    age_divided = pd.DataFrame(total_pop.groupby(['region_id', pd.cut(total_pop["age"], bins=bins+[110], labels=labels, include_lowest=True)]).sum('population')['population'])
    age_divided.reset_index(inplace=True)
    age_divided = age_divided.pivot(index='region_id', columns=['age'])['population']
    try:
        region_names_id = pd.read_csv(paths.municipalities_names, delimiter=",").drop_duplicates()
    except:
        region_names_id = pd.read_csv("../" + paths.municipalities_names, delimiter=",").drop_duplicates()
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
    N_j = df.contact_age_1.value_counts()[labels]
    N_eu = pd.read_csv(paths.europe_data)
    N_eu.age = pd.cut(N_eu['age'], bins=bins+[110], labels=labels, include_lowest=True)
    N_eu = N_eu.groupby('age').sum()['population']
    N_eu_tot = N_eu.sum()
    N_norway = population[population.columns[2:-1]].sum()
    N_norway_tot = np.sum(N_norway)

    contact_matrices = []
    for col in ['home', 'school', 'work', 'public']:
        M_ij = (pd.pivot_table(df_mat, values=col, index='contact_age_0', columns='contact_age_1').values)

        # Density scale transformation
        F_ij = M_ij / N_j.values.reshape(-1,1)

        # Density correction
        corrected_matrix = np.zeros((F_ij.shape))
        for i, a_i in enumerate(labels):
            for j, a_j in enumerate(labels):
                corrected_matrix[i][j] = F_ij[i][j] * (N_eu_tot * N_norway[a_j])/(N_eu[a_j] * N_norway_tot)

        # Symmetry
        symmetric_matrix = np.zeros((corrected_matrix.shape))
        for i, a_i in enumerate(labels):
            for j, a_j in enumerate(labels):
                symmetric_matrix[i][j] = (corrected_matrix[i][j] * N_norway[a_i] + corrected_matrix[j][i] * N_norway[a_j])/(N_norway[a_i]+N_norway[a_j])
        
        contact_matrices.append(symmetric_matrix)
    
    return contact_matrices

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
    df = pd.read_csv("data/age_groups/deaths_by_age.csv")
    df['age_group'] = pd.cut(df['age'], bins=bins+[110], labels=labels, include_lowest=True)
    return df.groupby('age_group').mean()['ifr'].to_numpy()/100

def get_historic_data():
    historic_data = pd.read_csv(paths.fhi_data_daily)  # set to None if not used
    historic_data.date = pd.to_datetime(historic_data.date)
    return historic_data
    
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
        datetime.date object with a given date
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
    
    data = np.array([age_labels, np.round(average_infected), np.round(average_vaccinated), np.round(average_dead), age_total]).T
    df = pd.DataFrame(columns=columns, data=data)
    for col in columns[1:]:
        df[col] = df[col].astype(float)
        df[col] = df[col].astype(int)
    total = df[df.columns[1:]].sum()
    total["Age group"] = "All"
    df = df.append(total, ignore_index=True)
    if save_to_file:
        df.to_csv(f"results/final_results_{policy}.csv", index=False)
    return df

def load_json(path):
    with open(path) as file:
        data = json.load(file)
    return data

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
    data = load_json(paths.wave_parameters)
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
        try:
            len_current_state = min([wave_state_timeline[::-1].index(state) for state in previous_states]) // decision_period_days
        except:
            len_current_state = 0
        wave_state_count = [wave_state_timeline[0]]
        for ws in wave_state_timeline[1:]:
            if ws != wave_state_count[-1]:
                wave_state_count.append(ws)
        from_start = False
    while True:
        n_wave = Counter(wave_state_count)[current_state]-1
        params = data['duration'][current_state][str(1 + n_wave%6)]
        duration = skewnorm.rvs(params['skew'], loc=params['mean'], scale=params['std'])
        duration = min(max(duration, params['min']), params['max']) // decision_period_days
        if not from_start: duration -= len_current_state
        try:
            for week in range(i, i+int(duration)):
                params = data['R'][current_state][str(1 + n_wave%6)]
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

def get_expected_yll(age_bins, age_labels):
    """ Retrieves the expected years remaining for each age group

    Args:
        age_bins (numpy.ndarray): int describing different age groups
        age_labels (numpy.ndarray): strings with description of different age groups
    Returns:
        int: yll
    """
    try:
        df = pd.read_csv(paths.expected_years)
    except:
        df = pd.read_csv("../" + paths.expected_years)
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
    for model_name in ['home', 'school', 'work', 'public', 'movement']:
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
    import geopandas as gpd
    pop['region_id'] = pop['region_id'].astype('str')
    pop['region_id'] = pop['region_id'].apply(lambda x: '{0:0>4}'.format(x))
    pop = pop[['region_id', 'population', 'region_name']]
    try:
        gdf = gpd.read_file(fpath_spatial_data)
    except:
        gdf = gpd.read_file("../" + fpath_spatial_data)
    gdf = gdf[['region_id', 'geometry']]
    df = pd.DataFrame(gdf)
    gdf = gpd.GeoDataFrame(df.merge(pop, right_on='region_id', left_on='region_id',  suffixes=('', '_y')), geometry='geometry')
    gdf = gdf.dropna()
    gdf = gdf.to_crs(3857)
    return gdf

def sort_filenames_by_date(files):
    dates = tuple(map(lambda x: x.split("_")[-6:], files))
    return sorted(tuple(map(lambda x: datetime(*map(int,x)), dates)))

def get_R_t(daily_cases):
    R_t = epyestim.estimate_r.estimate_r(
                                infections_ts = pd.Series(daily_cases),
                                gt_distribution = np.array([0,0,0,0,0,1]),
                                a_prior = 3,
                                b_prior = 1,
                                window_size = 7)
    for q in [0.025, 0.5, 0.975]:
        R_t[f'Q{q}'] = epyestim.estimate_r.gamma_quantiles(q, R_t['a_posterior'], R_t['b_posterior'])
    return R_t

def get_GA_params():
    run_from_file = bool(int(input("Run from file (bool): ")))
    if run_from_file:
        dir_path = "results/ga"
        files = os.listdir(dir_path)
        runs = dict(zip(range(1,len(files)+1),files))
        print(f"{tcolors.BOLD}Available runs:{tcolors.ENDC}")
        for k, v in runs.items(): print(f"{k}: {v}")
        file_nr = int(input("File (int): "))
        run = runs[file_nr]
        while True:
            try:
                gen = int(input("Run from generation: "))
                individuals_from_file = (gen, read_pickle(f'results/ga/{run}/individuals/individuals_{gen}.pkl'),
                                    read_pickle(f'results/ga/{run}/final_scores/final_score_{gen}.pkl'),
                                        read_pickle(f'results/ga/{run}/best_individuals/best_individual_{gen}.pkl'), run)
                params = load_json(f'results/ga/{run}/run_params.json')
            except FileNotFoundError:
                print(f"{tcolors.FAIL}Generation not available{tcolors.ENDC}")
                continue
            break
        print(f"{tcolors.OKGREEN}Running {runs[file_nr]} from generation {gen}{tcolors.ENDC}\nParams:")
        pprint(params)
        params["individuals_from_file"] = individuals_from_file
    else:
        instance_based = bool(int(input("Run instances (bool): ")))
        if instance_based:
            dir_path = "instances/"
            files = os.listdir(dir_path)
            instances = [load_json(dir_path + f) for f in sorted(files)]
            print(f"{tcolors.BOLD}Instances:{tcolors.ENDC}")
            for i, instance in enumerate(instances):
                print(f"{i+1}: Objective: {instance['objective']:>10}, Random individuals: {instance['random_individuals']}")
            selected = int(input("Run from instance: "))
            params = instances[selected-1]
            params['individuals_from_file'] = None
            print(f"{tcolors.OKGREEN}Running instance {selected}{tcolors.ENDC}\nParams:")
            pprint(params)
        else:
            params = {}
            params["gen"] = 0
            ga_objectives = {1: "fatalities", 2: "infected", 3: "weighted", 4: "yll"}
            print("Choose objective for genetic algorithm.")
            for k, v in ga_objectives.items(): print(f"{k}: {v}")
            ga_objective_number = int(input("\nGA Objective (int): "))
            params["objective"] = ga_objectives[ga_objective_number]
            params["random_individuals"] = bool(int(input("Random individual genes (bool): ")))
            params["population_size"] = int(input("Initial population size (int): "))
            params["simulations"] = int(input("Number of simulations (int): "))
            params["min_generations"] = int(input("Number of minimum generations (int): "))
            params["individuals_from_file"] = None
            print(f"{tcolors.OKGREEN}Running GA from start{tcolors.ENDC}\nParams:")
            pprint(params)
    return params

def write_csv(run_paths, folder_path, population, age_labels):
    print("Storing results ... ")
    S = np.array(list(map(lambda x: list(map(lambda y: y.S, x)), run_paths)))
    I = np.array(list(map(lambda x: list(map(lambda y: y.I, x)), run_paths)), dtype = object)
    new_infected = np.array(list(map(lambda x: list(map(lambda y: y.new_infected, x)), run_paths)), dtype = object)
    new_deaths = np.array(list(map(lambda x: list(map(lambda y: y.new_deaths, x)), run_paths)), dtype = object)
    vaccines_available = np.array(list(map(lambda x: list(map(lambda y: y.vaccines_available, x)), run_paths)), dtype = object)
    vaccinated = np.array(list(map(lambda x: list(map(lambda y: y.V, x)), run_paths)), dtype = object)
    contact_weights = np.array(list(map(lambda x: list(map(lambda y: y.contact_weights, x)), run_paths)), dtype = object)
    flow_scale = np.array(list(map(lambda x: list(map(lambda y: y.flow_scale, x)), run_paths)), dtype = object)
    dates = np.array(list(map(lambda x: list(map(lambda y: y.date, x)), run_paths)), dtype = object)
    num_sims, num_weeks = S.shape[0], S.shape[1]

    div_filepath = folder_path + "/div.csv"
    S_filepath = folder_path + "/S.csv" 
    I_filepath = folder_path + "/I.csv" 
    new_infected_filepath = folder_path + "/new_infected.csv" 
    new_deaths_filepath = folder_path + "/new_deaths.csv" 
    vaccinated_filepath = folder_path + "/vaccinated.csv" 
    
    identifying_columns = ["date", "simulation_nr", "week_nr"]
    div_columns = ["vaccines_available", "flow_scale", "contact_weight_1", "contact_weight_2", "contact_weight_3", "contact_weight_4"]
    S_region_columns = []
    I_region_columns = []
    new_infected_region_columns = []
    new_deaths_region_columns = []
    vaccinated_region_columns = []

    S_age_groups_columns = []
    I_age_groups_columns = []
    new_infected_age_groups_columns = []
    new_deaths_age_groups_columns = []
    vaccinated_age_groups_columns = []
    
    for i in range(population.shape[0]):
        S_region_columns += [f"S_region_{i+1}"]
        I_region_columns += [f"I_region_{i+1}"]
        new_infected_region_columns += [f"new_infected_region_{i+1}"]
        new_deaths_region_columns += [f"new_deaths_region_{i+1}"]
        vaccinated_region_columns += [f"vaccinated_region_{i+1}"]
    for i in range(len(age_labels)):
        S_age_groups_columns += [f"S_age_groups_{i+1}"]
        I_age_groups_columns += [f"I_age_groups_{i+1}"]
        new_infected_age_groups_columns += [f"new_infected_age_groups_{i+1}"]
        new_deaths_age_groups_columns += [f"new_deaths_age_groups_{i+1}"]
        vaccinated_age_groups_columns += [f"vaccinated_age_groups_{i+1}"]

    div_df = pd.DataFrame(columns=identifying_columns+div_columns) 
    S_df = pd.DataFrame(columns=identifying_columns+S_region_columns+S_age_groups_columns)
    I_df = pd.DataFrame(columns=identifying_columns+I_region_columns+I_age_groups_columns)
    new_infected_df = pd.DataFrame(columns=identifying_columns+new_infected_region_columns+new_infected_age_groups_columns)
    new_deaths_df = pd.DataFrame(columns=identifying_columns+new_deaths_region_columns+new_deaths_age_groups_columns)
    vaccinated_df = pd.DataFrame(columns=identifying_columns+vaccinated_region_columns+vaccinated_age_groups_columns)

    for i in tqdm(range(num_sims), ascii=True):
        for j in range(num_weeks):
            S_regions = list(S[i][j].sum(axis=1))
            I_regions = list(I[i][j].sum(axis=1))
            new_infected_regions = list(new_infected[i][j].sum(axis=1))
            new_deaths_regions = list(new_deaths[i][j].sum(axis=1))
            vaccinated_regions = list(vaccinated[i][j].sum(axis=1))

            S_age_groups = list(S[i][j].sum(axis=0))
            I_age_groups = list(I[i][j].sum(axis=0))
            new_infected_age_groups = list(new_infected[i][j].sum(axis=0))
            new_deaths_age_groups = list(new_deaths[i][j].sum(axis=0))
            vaccinated_age_groups = list(vaccinated[i][j].sum(axis=0))
            
            div_entry = {}
            S_entry = {}
            I_entry = {}
            new_infected_entry = {}
            new_deaths_entry = {}
            vaccinated_entry = {}
            identifying_data = [dates[i][j], i+1, j+1]
            div_data = identifying_data + [vaccines_available[i][j], flow_scale[i][j], contact_weights[i][j][0], contact_weights[i][j][1], contact_weights[i][j][2], contact_weights[i][j][3]]
            S_data = identifying_data + S_regions + S_age_groups
            I_data = identifying_data + I_regions + I_age_groups
            new_infected_data = identifying_data + new_infected_regions + new_infected_age_groups
            new_deaths_data = identifying_data + new_deaths_regions + new_deaths_age_groups
            vaccinated_data = identifying_data + vaccinated_regions + vaccinated_age_groups

            for k in range(len(div_df.columns)):
                div_entry[div_df.columns[k]] = div_data[k]
            for k in range(len(S_df.columns)):
                S_entry[S_df.columns[k]] = S_data[k]
                I_entry[I_df.columns[k]] = I_data[k]
                new_infected_entry[new_infected_df.columns[k]] = new_infected_data[k]
                new_deaths_entry[new_deaths_df.columns[k]] = new_deaths_data[k]
                vaccinated_entry[vaccinated_df.columns[k]] = vaccinated_data[k]
            
            div_df = div_df.append(div_entry, ignore_index=True) 
            S_df = S_df.append(S_entry, ignore_index=True)
            I_df = I_df.append(I_entry, ignore_index=True)
            new_infected_df = new_infected_df.append(new_infected_entry, ignore_index=True)
            new_deaths_df = new_deaths_df.append(new_deaths_entry, ignore_index=True)
            vaccinated_df = vaccinated_df.append(vaccinated_entry, ignore_index=True)

    div_df.to_csv(div_filepath)
    S_df.to_csv(S_filepath)
    I_df.to_csv(I_filepath)
    new_infected_df.to_csv(new_infected_filepath)
    new_deaths_df.to_csv(new_deaths_filepath)
    vaccinated_df.to_csv(vaccinated_filepath)

def read_csv(relative_path = "results/500_simulations_contact_based_2021_05_30_23_24_11"):
    dir_path = "../"
    folder_path = dir_path + relative_path
    div_filepath = folder_path + "/div.csv"
    S_filepath = folder_path + "/S.csv" 
    I_filepath = folder_path + "/I.csv" 
    new_infected_filepath = folder_path + "/new_infected.csv" 
    new_deaths_filepath = folder_path + "/new_deaths.csv" 
    vaccinated_filepath = folder_path + "/vaccinated.csv"
    pop_age_info_filepath = folder_path + "/start_date_population_age_labels.pkl"
    _, population, age_labels = read_pickle(pop_age_info_filepath)

    div_df = pd.read_csv(div_filepath, index_col=0)
    S_df = pd.read_csv(S_filepath, index_col=0)
    I_df = pd.read_csv(I_filepath, index_col=0)
    new_infected_df = pd.read_csv(new_infected_filepath, index_col=0)
    new_deaths_df = pd.read_csv(new_deaths_filepath, index_col=0)
    vaccinated_df = pd.read_csv(vaccinated_filepath, index_col=0)

    nr_simulations = div_df.loc[len(div_df)-1].simulation_nr
    nr_weeks = div_df.loc[len(div_df)-1].week_nr
    nr_regions = len(population)
    nr_age_groups = len(age_labels)

    vaccines_available = np.zeros((nr_simulations, nr_weeks))
    flow_scale = np.zeros((nr_simulations, nr_weeks))
    contact_weights = np.zeros((nr_simulations, nr_weeks, 4))
    for i in tqdm(range(nr_simulations)):
        vaccines_available[i, :] = div_df.loc[(i)*nr_weeks:(i+1)*nr_weeks -1].to_numpy()[:,3]
        flow_scale[i, :] = div_df.loc[(i)*nr_weeks:(i+1)*nr_weeks-1].to_numpy()[:,4] 
        contact_weights[i, :, :] = div_df.loc[(i)*nr_weeks:(i+1)*nr_weeks-1].to_numpy()[:,5:]

    S_regions = np.zeros((nr_simulations, nr_weeks, nr_regions))
    I_regions = np.zeros((nr_simulations, nr_weeks, nr_regions))
    new_infected_regions = np.zeros((nr_simulations, nr_weeks, nr_regions))
    new_deaths_regions = np.zeros((nr_simulations, nr_weeks, nr_regions))
    vaccinated_regions = np.zeros((nr_simulations, nr_weeks, nr_regions))
    S_age_groups = np.zeros((nr_simulations, nr_weeks, nr_age_groups))
    I_age_groups = np.zeros((nr_simulations, nr_weeks, nr_age_groups))
    new_infected_age_groups = np.zeros((nr_simulations, nr_weeks, nr_age_groups))
    new_deaths_age_groups = np.zeros((nr_simulations, nr_weeks, nr_age_groups))
    vaccinated_age_groups = np.zeros((nr_simulations, nr_weeks, nr_age_groups))
    for i in tqdm(range(nr_simulations)):
        S_regions[i, :, :] = S_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,3:-7]
        S_age_groups[i, :, :] = S_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,-7:]
        I_regions[i, :, :] = I_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,3:-7]
        I_age_groups[i, :, :] = I_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,-7:]
        new_infected_regions[i, :, :] = new_infected_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,3:-7]
        new_infected_age_groups[i, :, :] = new_infected_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,-7:]
        new_deaths_regions[i, :, :] = new_deaths_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,3:-7]
        new_deaths_age_groups[i, :, :] = new_deaths_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,-7:]
        vaccinated_regions[i, :, :] = vaccinated_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,3:-7]
        vaccinated_age_groups[i, :, :] = vaccinated_df.loc[(i)*nr_weeks:(i+1)*nr_weeks - 1].to_numpy()[:,-7:]
    
    S_df['date'] = pd.to_datetime(S_df['date'], format='%Y-%m-%d')
    dates = S_df['date'].iloc[:nr_weeks]
    
    return (age_labels,
            vaccines_available, 
            flow_scale,
            contact_weights,
            S_regions,
            I_regions,
            new_infected_regions,
            new_deaths_regions,
            vaccinated_regions,
            S_age_groups,
            I_age_groups,
            new_infected_age_groups,
            new_deaths_age_groups,
            vaccinated_age_groups,
            dates)
