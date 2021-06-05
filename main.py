import plot
import utils
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from vaccine_allocation_model.GA import SimpleGeneticAlgorithm, Individual
from vaccine_allocation_model.Policy import Policy
from vaccine_allocation_model.SEAIR import SEAIR
import numpy as np
from pandas import Timedelta
from tqdm import tqdm
from datetime import datetime
import os
np.random.seed(42)

if __name__ == '__main__':
    # Set initial parameters
    runs = 500
    decision_period = 56
    start_day, start_month, start_year = 24, 2, 2020
    start_date = utils.get_date(f"{start_year}{start_month:02}{start_day:02}")
    end_day, end_month, end_year = 31, 7, 2021
    end_date = utils.get_date(f"{end_year}{end_month:02}{end_day:02}")
    horizon = int(Timedelta(end_date-start_date).days // (decision_period/4))
    initial_infected = 10
    policies = ['random', 'no_vaccines', 'susceptible_based', 
                'infection_based', 'oldest_first', 'contact_based', 
                'weighted', 'fhi_policy']
    policy_number = int(input("choose policy: "))
    individual = Individual()
    weights = individual.genes

    # Read data and generate parameters
    paths = utils.create_named_tuple('paths', 'filepaths.txt')
    config = utils.create_named_tuple('config', paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    expected_years_remaining = utils.get_expected_yll(config.age_bins, age_labels)
    commuters = utils.generate_commuter_matrix(age_group_flow_scaling)
    response_measure_model = utils.load_response_measure_models()
    historic_data = utils.get_historic_data()
    
    # Run settings
    run_GA = False
    include_flow = True
    stochastic = True
    use_wave_factor = True
    use_response_measures = True
    verbose = False
    plot_results = False
    plot_geo = False
    write_simulations_to_file = True

    vaccine_policy = Policy(
                    config=config,
                    policy='weighted' if run_GA else policies[policy_number],
                    population=population[population.columns[2:-1]].values,
                    contact_matrices=contact_matrices,
                    age_flow_scaling=age_group_flow_scaling,
                    GA=True)

    epidemic_function = SEAIR(
                    commuters=commuters,
                    contact_matrices=contact_matrices,
                    population=population,
                    age_group_flow_scaling=age_group_flow_scaling,
                    death_rates=death_rates,
                    config=config,
                    include_flow=include_flow,
                    stochastic=stochastic,
                    use_wave_factor=use_wave_factor)

    initial_state = State.generate_initial_state(
                    num_initial_infected=initial_infected,
                    contact_weights=config.initial_contact_weights,
                    flow_scale=config.initial_flow_scale,
                    population=population,
                    start_date=start_date)

    mdp = MarkovDecisionProcess(
                    config=config,
                    decision_period=decision_period,
                    population=population, 
                    epidemic_function=epidemic_function,
                    initial_state=initial_state,
                    response_measure_model=response_measure_model, 
                    use_response_measures=use_response_measures,
                    use_wave_factor=use_wave_factor,
                    horizon=horizon,
                    end_date=end_date,
                    policy=vaccine_policy,
                    historic_data=historic_data,
                    verbose=verbose)

    if run_GA:
        params = utils.get_GA_params()
        GA = SimpleGeneticAlgorithm(
                simulations=params["simulations"], 
                population_size=params["population_size"], 
                process=mdp,
                objective=params["objective"],
                min_generations=params["min_generations"],
                random_individuals=params["random_individuals"],
                expected_years_remaining=expected_years_remaining,
                verbose=True,
                individuals_from_file=params["individuals_from_file"])
        GA.run()
    else:
        weighted = policies[policy_number] == 'weighted'
        print(f"Running {policies[policy_number] if not weighted else 'GA_individual'} policy ({runs} simulations).")
        print(f"Storing results to csv is set to {str(write_simulations_to_file)}.")
        if weighted: 
            print(f"Genes: ")
            for weight in weights:
                print()
                print(weight)
        results = []
        run_paths = []
        seeds = np.arange(runs)
        for i in tqdm(range(runs)):
            np.random.seed(seeds[i])
            mdp.init()
            mdp.reset()
            mdp.run(weights)
            results.append(mdp.state)
            while len(mdp.path) < horizon+1: # Ensure all paths are equal length
                mdp.path.append(mdp.state)
            run_paths.append(mdp.path)
            utils.print_results(mdp.state, population, age_labels, vaccine_policy)
            print("\n",mdp.state.trend_count,"\n")

        avg_results = utils.get_average_results(results, population, age_labels, vaccine_policy)
        
        if write_simulations_to_file:
            start_of_run = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            run_folder = f"/results/{runs}_simulations_{policies[policy_number]}_{start_of_run}"
            folder_path = os.getcwd() + run_folder
            start_date_population_age_labels_path = folder_path + "/start_date_population_age_labels.pkl"
            os.mkdir(folder_path)
            utils.write_pickle(start_date_population_age_labels_path, [start_date, population, age_labels])
            utils.write_csv(run_paths, folder_path, population, age_labels)

    if plot_results and not run_GA:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        results_age = history.sum(axis=2)
        results_regions = history.sum(axis=3)
        infection_results_age = new_infections.sum(axis=1)
        infection_results_regions = new_infections.sum(axis=2)
        regions_to_plot = ['OSLO', 'TRONDHEIM', 'LØRENSKOG', 'STEINKJER']
        comps_to_plot = ["E2", "A", "I"]

        if use_response_measures:
            plot.plot_control_measures(mdp.path, all=True)
        if use_wave_factor:
            R_eff = mdp.wave_timeline
            plot.age_group_infected_plot_weekly(results_age, start_date, age_labels, R_eff, include_R=True)
        else:
            plot.age_group_infected_plot_weekly(results_age, start_date, age_labels, include_R=False)
        plot.plot_R_t(epidemic_function.daily_cases)
        #plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        #plot.seir_plot_weekly_several_regions(results_regions, start_date, comps_to_plot, regions_to_plot, paths.municipalities_names)
        #plot.infection_plot_weekly_several_regions(infection_results_regions, start_date, regions_to_plot, paths.municipalities_names)#

    if plot_geo:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        plot.plot_geospatial(paths.municipalities_geo, history, paths.municipality_plots, population, accumulated_compartment_plot=False, per_100k=False)
        plot.create_gif(paths.municipality_gif, paths.municipality_plots)
        plot.plot_commuters(population, paths.municipalities_geo, paths.municipalities_commuters)
        plot.plot_norway_map(population, paths.municipalities_geo)
        plot.plot_population()