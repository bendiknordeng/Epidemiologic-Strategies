from covid import plot
from covid import utils
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.SEAIR import SEAIR
from vaccine_allocation_model.GA import SimpleGeneticAlgorithm
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # Get filepaths 
    paths = utils.create_named_tuple('filepaths.txt')

    # Set initial parameters
    np.random.seed(10)
    day = 24
    month = 2
    year = 2020
    runs = 1
    start_date = utils.get_date(f"{year}{month:02}{day:02}")
    horizon = 60 # number of decision_periods
    decision_period = 28
    initial_infected = 1000
    initial_vaccines_available = 0
    policies = ['random', 'no_vaccines', 'susceptible_based', 'infection_based', 'oldest_first', 'weighted']
    policy = policies[-2]
    weighted_policy_weights = [0, 0.33, 0.33, 0.34]
    initial_wave_state = 'U'
    initial_wave_count = {'U': 1, 'D': 0, 'N': 0}

    # Read data and generate parameters
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    OD_matrices = utils.generate_ssb_od_matrix(population, age_group_flow_scaling, paths.municipalities_commute)
    response_measure_model = utils.load_response_measure_models()
    historic_data = utils.get_historic_data(paths.fhi_data_daily)

    # Simulation settings
    verbose = False
    use_response_measures = False
    include_flow = True
    stochastic = False

    plot_results = True
    write_weekly = False
    write_to_csv = False 

    epidemic_function = SEAIR(
                        OD=OD_matrices,
                        contact_matrices=contact_matrices,
                        population=population,
                        age_group_flow_scaling=age_group_flow_scaling,
                        death_rates=death_rates,
                        config=config,
                        paths=paths,
                        write_to_csv=write_to_csv, 
                        write_weekly=write_weekly,
                        include_flow=include_flow,
                        stochastic=stochastic)

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
                        horizon=horizon,
                        policy=policy,
                        historic_data=historic_data,
                        verbose=verbose)

    results = []                   
    for i in tqdm(range(runs)):
        mdp.run()
        results.append(mdp.path[-1])
        utils.print_results(mdp.path[-1], population, age_labels, policy)
        if i != (runs-1): mdp.reset()
    utils.get_average_results(results, population, age_labels, policy)

    # GA = SimpleGeneticAlgorithm(3, mdp)
    
    # while not GA.converged:
    #     GA.new_generation()
    #     GA.find_fitness(runs)
    #     GA.evaluate_fitness()

    
    history, new_infections = utils.transform_path_to_numpy(mdp.path)
        
    
    # history shape == (61, 8, 356, 6)
    # hist_accumulated_by_age.shape == (61, 8, 356)
    # history, new_infections = utils.transform_path_to_numpy(mdp.path)
    # hist_accumulated_by_age = np.sum(history, axis=3)
    # hist_infected_accumulated_by_age = hist_accumulated_by_age[:, 1, :]

    # import pdb; pdb.set_trace()
    
    # regions_with_inf = []

    
    # regions_with_infections = []
    # for w in range(len(hist_accumulated_by_age.shape(axis=0))):
    #     region_hist_infected_t
    #     np.argwhere(x > 0.01)


    if plot_results:
        history, new_infections = utils.transform_path_to_numpy(mdp.path)
        #plot.plot_control_measures(mdp.path, all=False)

        results_age = history.sum(axis=2)
        plot.age_group_infected_plot_weekly(results_age, start_date, age_labels)
        infection_results_age = new_infections.sum(axis=1)
        plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        
        utils.get_r_effective(mdp.path, population, config, from_data=False)

