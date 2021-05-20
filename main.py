import plot
import utils
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from vaccine_allocation_model.GA import SimpleGeneticAlgorithm
from vaccine_allocation_model.Policy import Policy
from vaccine_allocation_model.SEAIR import SEAIR
import numpy as np
from pandas import Timedelta
from tqdm import tqdm

if __name__ == '__main__':
    # Set initial parameters
    runs = 50
    decision_period = 28
    start_day, start_month, start_year = 24, 2, 2020
    start_date = utils.get_date(f"{start_year}{start_month:02}{start_day:02}")
    end_day, end_month, end_year = 31, 7, 2021
    end_date = utils.get_date(f"{end_year}{end_month:02}{end_day:02}")
    horizon = int(Timedelta(end_date-start_date).days // (decision_period/4))
    initial_infected = 100
    initial_vaccines_available = 0
    policies = ['random', 'no_vaccines', 'susceptible_based', 
                'infection_based', 'oldest_first', 'contact_based', 
                'commuter_based', 'weighted', 'fhi_policy']
    policy_number = -3
    weights = np.array([0, 0, 0, 1, 0, 0])

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
    run_GA = True
    include_flow = True
    use_waves = True
    stochastic = True
    use_response_measures = False
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
        results = []
        mdp.init()
        for i in tqdm(range(runs)):
            np.random.seed(i*10)
            mdp.reset()
            mdp.run(weights)
            results.append(mdp.state)
            utils.print_results(mdp.state, population, age_labels, vaccine_policy)
        utils.get_average_results(results, population, age_labels, vaccine_policy)

    if plot_results:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        R_eff = mdp.wave_timeline
        results_age = history.sum(axis=2)
        results_regions = history.sum(axis=3)
        infection_results_age = new_infections.sum(axis=1)
        infection_results_regions = new_infections.sum(axis=2)
        regions_to_plot = ['OSLO', 'TRONDHEIM', 'LØRENSKOG', 'STEINKJER']
        comps_to_plot = ["E2", "A", "I"]

        plot.age_group_infected_plot_weekly(results_age, start_date, age_labels, R_eff, include_R=True)
        plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        utils.get_r_effective(mdp.path, population, config, from_data=False)
        plot.seir_plot_weekly_several_regions(results_regions, start_date, comps_to_plot, regions_to_plot, paths.municipalities_names)
        plot.infection_plot_weekly_several_regions(infection_results_regions, start_date, regions_to_plot, paths.municipalities_names)

    if plot_geo:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        plot.plot_geospatial(paths.municipalities_geo, history, paths.municipality_plots, population, accumulated_compartment_plot=False, per_100k=False)
        plot.create_gif(paths.municipality_gif, paths.municipality_plots)
        plot.plot_commuters(population, paths.municipalities_geo, paths.municipalities_commuters)