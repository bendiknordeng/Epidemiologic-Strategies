from collections import defaultdict
from covid import plot
from covid import utils
import numpy as np
import pandas as pd
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.seair import SEAIR
from vaccine_allocation_model.GA import SimpleGeneticAlgorithm
from tqdm import tqdm

if __name__ == '__main__':
    # Get filepaths 
    paths = utils.create_named_tuple('filepaths.txt')

    # Set initial parameters
    day = 21
    month = 2
    year = 2020
    runs = 2
    start_date = utils.get_date(f"{year}{month:02}{day:02}")
    horizon = 60 # number of weeks
    decision_period = 28
    initial_infected = 5
    initial_vaccines_available = 0
    policies = ['random', 'no_vaccines', 'susceptible_based', 'infection_based', 'oldest_first', 'weighted']
    policy = policies[-2]
    plot_results = True
    verbose = False
    weighted_policy_weights = [0, 0.33, 0.33, 0.34]
    use_response_measures = False
    initial_wave_state = 'U'
    initial_wave_count = {'U': 1, 'D': 0, 'N': 0}

    # Read data and generate parameters
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    OD_matrices = utils.generate_ssb_od_matrix(decision_period, population, paths.municipalities_commute)
    response_measure_model = utils.load_response_measure_models()
    wave_timeline, wave_state_timeline = utils.get_wave_timeline(horizon)
    historic_data = utils.get_historic_data(paths.fhi_data_daily)

    epidemic_function = SEAIR(
                        OD=OD_matrices,
                        contact_matrices=contact_matrices,
                        population=population,
                        age_group_flow_scaling=age_group_flow_scaling,
                        death_rates=death_rates,
                        config=config,
                        paths=paths,
                        write_to_csv=False, 
                        write_weekly=False,
                        include_flow=True,
                        stochastic=True)

    initial_state = State.initialize_state(
                        num_initial_infected=initial_infected,
                        vaccines_available=initial_vaccines_available,
                        contact_weights=config.initial_contact_weights,
                        alphas=config.initial_alphas,
                        flow_scale=config.initial_flow_scale,
                        population=population,
                        wave_state=initial_wave_state,
                        wave_count=initial_wave_count,
                        start_date=start_date)
    
    mdp = MarkovDecisionProcess(
                        config=config,
                        decision_period=decision_period,
                        population=population, 
                        epidemic_function=epidemic_function,
                        initial_state=initial_state,
                        response_measure_model=response_measure_model, 
                        use_response_measures=use_response_measures, 
                        wave_timeline=wave_timeline, 
                        wave_state_timeline=wave_state_timeline,
                        horizon=horizon,
                        policy=policy,
                        historic_data=historic_data,
                        verbose=verbose)
    # mdp.run()
    # utils.print_results(mdp.path[-1], population, age_labels, policy)

    GA = SimpleGeneticAlgorithm(3, mdp)
    
    while not GA.converged:
        GA.new_generation()
        GA.find_fitness(runs)
        GA.evaluate_fitness()

    if plot_results:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        #plot.plot_control_measures(mdp.path, all=False)

        results_age = history.sum(axis=2)
        plot.age_group_infected_plot_weekly(results_age, start_date, age_labels)
        infection_results_age = new_infections.sum(axis=1)
        plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        
        #utils.get_r_effective(mdp.path, population, config, from_data=False)

