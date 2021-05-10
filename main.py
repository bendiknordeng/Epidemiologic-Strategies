from covid import plot
from covid import utils
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from vaccine_allocation_model.GA import SimpleGeneticAlgorithm
from vaccine_allocation_model.Policy import Policy
from covid.SEAIR import SEAIR
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    # Get filepaths 
    paths = utils.create_named_tuple('filepaths.txt')

    # Set initial parameters
    np.random.seed(10)
    runs = 1
    day = 30
    month = 4
    year = 2020
    start_date = utils.get_date(f"{year}{month:02}{day:02}")
    horizon = 60 # number of decision_periods
    decision_period = 28
    initial_infected = 10
    initial_vaccines_available = 0
    policies = ['random', 'no_vaccines', 'susceptible_based', 'infection_based', 'oldest_first', 'weighted']
    policy_number = -2
    ga_objectives = ["deaths", "weighted", "yll"]
    ga_objective_number = -1

    # Read data and generate parameters
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    commuters = utils.generate_commuter_matrix(age_group_flow_scaling, paths.municipalities_commute)
    response_measure_model = utils.load_response_measure_models()
    historic_data = utils.get_historic_data(paths.fhi_data_daily)
    
    # Simulation settings
    run_GA = False
    verbose = False
    use_response_measures = False
    include_flow = True
    use_waves = True
    stochastic = True
    plot_results = False
    plot_geo = True


    vaccine_policy = Policy(
                    config=config,
                    policy=policies[policy_number],
                    population=population[population.columns[2:-1]].values)

    epidemic_function = SEAIR(
                    commuters=commuters,
                    contact_matrices=contact_matrices,
                    population=population,
                    age_group_flow_scaling=age_group_flow_scaling,
                    death_rates=death_rates,
                    config=config,
                    include_flow=include_flow,
                    stochastic=stochastic,
                    use_waves=use_waves)

    initial_state = State.generate_initial_state(
                    num_initial_infected=initial_infected,
                    vaccines_available=initial_vaccines_available,
                    contact_weights=config.initial_contact_weights,
                    alphas=config.initial_alphas,
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
                    horizon=horizon,
                    policy=vaccine_policy,
                    historic_data=historic_data,
                    verbose=verbose)

    if run_GA:
        GA = SimpleGeneticAlgorithm(runs, 20, mdp, ga_objectives[ga_objective_number], verbose=True)
        GA.run()
    else:
        results = []                   
        for i in tqdm(range(runs)):
            np.random.seed(i*10)
            mdp.init()
            mdp.run()
            results.append(mdp.path[-1])
            utils.print_results(mdp.path[-1], population, age_labels, vaccine_policy)
        utils.get_average_results(results, population, age_labels, vaccine_policy)


    if plot_results:
        # import pdb; pdb.set_trace()
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        
        R_eff = mdp.wave_timeline
        results_age = history.sum(axis=2)
        plot.age_group_infected_plot_weekly(results_age, start_date, age_labels, R_eff, include_R=True)
        # infection_results_age = new_infections.sum(axis=1)
        # plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        
        # utils.get_r_effective(mdp.path, population, config, from_data=False)
        #plot.plot_control_measures(mdp.path, all=False)
        

    if plot_geo:
        fpath = 'data/geospatial/municipalities_spatial_data.json'
        gdf = utils.generate_geopandas(population, fpath)
        history, new_infections = utils.transform_path_to_numpy(mdp.path) 
        res = history.sum(axis=3) #weeks, #compartments, #regions
        plot.plot_spatial(gdf, res)