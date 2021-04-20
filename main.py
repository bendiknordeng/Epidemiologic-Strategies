from covid import plot
from covid import utils
import numpy as np
import pandas as pd
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.seair import SEAIR

if __name__ == '__main__':
    # Get filepaths 
    paths = utils.create_named_tuple('filepaths.txt')
    
    # Read data and generate parameters
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, population, country='GB')
    plot.plot_heatmaps(contact_matrices, [0.15,0.3,0.3,0.05,0.2], age_labels, paths.heat_maps)
    age_group_flow_scaling = utils.get_age_group_flow_scaling(config.age_bins, age_labels, population)
    death_rates = utils.get_age_group_fatality_prob(config.age_bins, age_labels)
    OD_matrices = utils.generate_ssb_od_matrix(28, population, paths.municipalities_commute)
    historic_data = utils.get_historic_data(paths.fhi_data_daily)
    population.to_csv('data/temp_pop.csv', index=False)
    policies = ['no_vaccines', 'population_based', 'susceptible_based', 'infection_based', 'adults_first', 'oldest_first']

    # Set initial parameters
    np.random.seed(10)
    day = 1
    month = 12
    year = 2020
    start_date = utils.get_date(f"{year}{month:02}{day:02}")
    horizon = 60 # number of weeks
    decision_period = 28
    for i in ['no_vaccines', 'population_based', 'susceptible_based', 'infection_based', 'adults_first', 'oldest_first']:
        policy = i
        initial_infected = 1
        initial_vaccines_available = 0
        plot_results = False

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
                            hidden_cases=True)

        initial_state = State.initialize_state(
                            num_initial_infected=initial_infected,
                            vaccines_available=initial_vaccines_available, 
                            population=population,
                            start_date=start_date)

        mdp = MarkovDecisionProcess( 
                            population=population, 
                            epidemic_function=epidemic_function,
                            initial_state=initial_state,
                            horizon=horizon, 
                            decision_period=28, 
                            policy=policy,
                            historic_data=historic_data)

        path = mdp.run(verbose=False)

        history, new_infections = utils.transform_path_to_numpy(path)
        utils.print_results(history, new_infections, population, age_labels, policy, save_to_file=False)
        if plot_results:
            results_age = history.sum(axis=2)
            plot.age_group_infected_plot_weekly(results_age, start_date, age_labels)
            infection_results_age = new_infections.sum(axis=1)
            plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
            
            results_compartment = history.sum(axis=3).sum(axis=2)
            labels = ['S', 'E1', 'E2', 'A', 'I', 'R', 'D', 'V']
            plot.seir_plot_weekly(results_compartment, start_date, labels)

            # load necessary data for SEIR development plot
            # df = pd.read_csv(paths.results_history)
            # history = utils.transform_df_to_history(df, 'SEAIQRDVH', 356, 5)
            # results = history.sum(axis=2)
            
            # plot.plot_heatmaps(contact_matrices, [0.5, 0.5, 0.5, 0.5, 0.5], age_labels, paths.heat_maps)

            # accumulated SEIR development plot
            # plot.seir_plot(results)

                # load necessary data for geospatial plot 
                # kommuner_geometry = plot.create_geopandas(True, population, paths.municipalities_geo_pkl, paths.municipalities_geo_geojson)

                # geospatial plot of simulation
                # plot.plot_simulation(history[::4,:,:], population, results[::4,4], kommuner_geometry, paths.municipality_plots)
                # plot.create_gif(paths.municipality_gif,paths.municipality_plots)

                # geospatial plot of of historical covid cases in Norway
                # history = utils.transform_historical_df_to_history(pd.read_csv(paths.municipalities_hist_data))
                # plot.plot_historical_infected(history[::1,:,:], population, kommuner_geometry, paths.municipality_hist_plots)
                # plot.create_gif(paths.municipality_hist_gif,paths.municipality_hist_plots)

