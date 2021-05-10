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
    runs = 5
    day = 21
    month = 2
    year = 2020
    start_date = utils.get_date(f"{year}{month:02}{day:02}")
    horizon = 70 # number of decision_periods
    decision_period = 28
    initial_infected = 10
    initial_vaccines_available = 0
    policies = ['random', 'no_vaccines', 'susceptible_based', 
                'infection_based', 'oldest_first', 'contact_based', 
                'commuter_based', 'weighted']
    weights = np.array([0, 0, 0, 0, 0, 1])
    policy_number = -1

    # Read data and generate parameters
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    expected_years_remaining = utils.get_expected_yll(config.age_bins, age_labels)
    commuters = utils.generate_commuter_matrix(age_group_flow_scaling, paths.municipalities_commute)
    response_measure_model = utils.load_response_measure_models()
    historic_data = utils.get_historic_data(paths.fhi_data_daily)

    # Run settings
    run_GA = True
    use_response_measures = False
    include_flow = True
    use_waves = True
    stochastic = True
    verbose = False
    plot_results = False
    plot_geo = False

    vaccine_policy = Policy(
                    config=config,
                    policy=policies[policy_number],
                    population=population[population.columns[2:-1]].values,
                    contact_matrices=contact_matrices,
                    age_flow_scaling=age_group_flow_scaling,
                    GA=run_GA)

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
        ga_objectives = {1: "deaths", 2: "weighted", 3: "yll"}
        print("Choose objective for genetic algorithm.")
        for k, v in ga_objectives.items(): print(f"{k}: {v}")
        ga_objective_number = int(input("\nGA Objective (int): "))
        random_individuals = bool(input("Random individual genes (bool): "))
        population_size = int(input("Initial population size (int): "))
        simulations = int(input("Number of simulations (int): "))

        GA = SimpleGeneticAlgorithm(
                    simulations=simulations, 
                    population_size=population_size, 
                    process=mdp,
                    objective=ga_objectives[ga_objective_number], 
                    random_individuals=random_individuals,
                    expected_years_remaining=expected_years_remaining,
                    verbose=True)
        GA.run()
    else:
        results = []                   
        for i in tqdm(range(runs)):
            np.random.seed(i*10)
            mdp.reset()
            mdp.run(weights)
            results.append(mdp.state)
            utils.print_results(mdp.state, population, age_labels, vaccine_policy)
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
        # plot.plot_control_measures(mdp.path, all=False)   

    if plot_geo:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        history_age_accumulated = history.sum(axis=3)

        # plot seir for different regions
        comps_to_plot = ["E1", "E2", "A", "I", "R", "D", "V"]
        regions_to_plot = ['OSLO', 'BÅTSFJORD']             
        plot.seir_plot_weekly_several_regions(history_age_accumulated, start_date, comps_to_plot, regions_to_plot, paths.municipalities_names)

        # plot geospatial data
        gdf = utils.generate_geopandas(population, paths.municipalities_geo)
        plot.plot_geospatial(gdf, history_age_accumulated, paths.municipality_plots)
        
        # generate gif
        plot.create_gif(paths.municipality_gif, paths.municipality_plots)