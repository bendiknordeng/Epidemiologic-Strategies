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
    age_bins = [0,15,20,40,66]
    age_labels = utils.generate_labels_from_bins(age_bins)
    population = utils.generate_custom_population(age_bins, age_labels, paths.age_divided_population, paths.municipalities_names)
    OD_matrices = utils.generate_ssb_od_matrix(28, population, paths.municipalities_commute)

    historic_data = pd.read_csv(paths.fhi_data_daily)  # set to None if not used
    historic_data.date = pd.to_datetime(historic_data.date)
    
    epidemic_function = SEAIR(
                OD=OD_matrices,
                population=population,
                time_delta=config.time_delta,
                contact_matrices=config.contact_matrices,
                age_group_flow_scaling=config.age_group_flow_scaling,
                R0=config.R0,
                efficacy=config.efficacy,
                latent_period=config.latent_period, 
                proportion_symptomatic_infections=config.proportion_symptomatic_infections,
                presymptomatic_infectiousness=config.presymptomatic_infectiousness,
                asymptomatic_infectiousness=config.asymptomatic_infectiousness,
                presymptomatic_period=config.presymptomatic_period,
                postsymptomatic_period=config.postsymptomatic_period,
                fatality_rate_symptomatic=config.fatality_rate_symptomatic,
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
                        vaccines_available=0, 
                        population=population,
                        start_date=start_date,
                        time_step=0)

    horizon = 60 # number of weeks
    policies = ['no_vaccines', 'random', 'population_based', 'infection_based']
    mdp = MarkovDecisionProcess( 
                    population=population, 
                    epidemic_function=epidemic_function,
                    initial_state=initial_state,
                    horizon=horizon, 
                    decision_period=28, 
                    policy=policies[2],
                    historic_data=historic_data,
                    verbose=False)

    path = mdp.run()
    history, new_infections = utils.transform_path_to_numpy(path)
    utils.print_results(history, new_infections, population, age_labels, save_to_file=True)

    results_age = history.sum(axis=2)
    plot.age_group_infected_plot_weekly(results_age, start_date, age_labels)
    #infection_results_age = new_infections.sum(axis=1)
    #plot.age_group_infected_plot_weekly_cumulative(infection_results_age, start_date, age_labels)
    
    results_compartment = history.sum(axis=3).sum(axis=2)
    labels= ['S', 'E1', 'E2', 'A', 'I', 'R', 'D', 'V']
    plot.seir_plot_weekly(results_compartment, start_date, labels)

    # plot confusion matrices
    # plot.plot_heatmaps(config.contact_matrices, config.contact_matrices_weights, paths.heat_maps)

    # load necessary data for SEIR development plot
    # df = pd.read_csv(paths.results_history)
    # history = utils.transform_df_to_history(df, 'SEAIQRDVH', 356, 5)
    # results = history.sum(axis=2)
    
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

