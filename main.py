from covid import plot
from covid import utils
import numpy as np
import pandas as pd
from vaccine_allocation_model.State import State
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.seair import SEAIR

if __name__ == '__main__':
    # read filepaths 
    paths = utils.create_named_tuple('filepaths.txt')
    
    # read in data from filepaths 
    config = utils.create_named_tuple(paths.config)
    age_labels = utils.generate_labels_from_bins(config.age_bins)
    population = utils.generate_custom_population(config.age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    contact_matrices = utils.generate_contact_matrices(config.age_bins, age_labels, country='GB')
    OD_matrices = utils.generate_ssb_od_matrix(28, population, paths.municipalities_commute)
    historic_data = utils.get_historic_data(paths.fhi_data_daily)
    
    epidemic_function = SEAIR(
                        OD=OD_matrices,
                        contact_matrices=contact_matrices,
                        population=population,
                        config=config,
                        paths=paths,
                        write_to_csv=False, 
                        write_weekly=False,
                        include_flow=False,
                        hidden_cases=False)

    # Set start date
    day = 21
    month = 2
    year = 2020
    start_date = utils.get_date(f"{year}{month:02}{day:02}")

    initial_state = State.initialize_state(
                        num_initial_infected=1000,
                        vaccines_available=1000, 
                        population=population,
                        start_date=start_date,
                        time_step=0)

    horizon = 60 # number of weeks
    policy = {
        0: 'no_vaccines', 
        1: 'random', 
        2: 'population_based', 
        3: 'infection_based',
        4: 'age_based'
        }[0]
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
    plot_results = False
    if plot_results:
        results_age = history.sum(axis=2)
        plot.age_group_infected_plot_weekly(results_age, start_date, age_labels)
        infection_results_age = new_infections.sum(axis=1)
        plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
        
        results_compartment = history.sum(axis=3).sum(axis=2)
        labels= ['S', 'E1', 'E2', 'A', 'I', 'R', 'D', 'V']
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

