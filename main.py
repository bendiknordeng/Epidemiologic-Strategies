from covid import plot
from covid import utils
import numpy as np
import pandas as pd
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.seaiqr import SEAIQR

if __name__ == '__main__':
    # read filepaths 
    paths = utils.create_named_tuple('filepaths.txt')
    
    # read in data from filepaths 
    config = utils.create_named_tuple(paths.config)
    population = utils.create_population(paths.age_divided_population, paths.municipalities_names)
    OD_matrices = utils.generate_ssb_od_matrix(28, population, paths.municipalities_commute)
    
    #vaccine_supply = read_pickle(paths.municipalities_v)
    vaccine_supply = np.ones((28,356))
    epidemic_function = SEAIQR(OD_matrices,
                population,
                contact_matrices=config.contact_matrices,
                age_group_flow_scaling=config.age_group_flow_scaling,
                contact_matrices_weights=config.contact_matrices_weights,
                R0=config.R0,
                efficacy=config.efficacy,
                proportion_symptomatic_infections=config.proportion_symptomatic_infections,
                latent_period=config.latent_period*config.periods_per_day, 
                recovery_period=config.recovery_period*config.periods_per_day,
                pre_isolation_infection_period=config.pre_isolation_infection_period*config.periods_per_day, 
                post_isolation_recovery_period=config.post_isolation_recovery_period*config.periods_per_day, 
                fatality_rate_symptomatic=config.fatality_rate_symptomatic)

    # run simulation
    horizon = 58 # number of weeks
    mdp = MarkovDecisionProcess(OD_matrices, population, epidemic_function, vaccine_supply, horizon, decision_period=28, policy="random")
    path = mdp.run()

    history = utils.transform_path_to_numpy(path)
    results = history.sum(axis=2)
    plot.age_group_infected_plot_weekly(results)

    results = history.sum(axis=3).sum(axis=2)
    plot.seir_plot_weekly(results)

    # load necessary data for SEIR development plot
    """ df = pd.read_csv(paths.results_history)
    history = utils.transform_df_to_history(df, 'SEAIQRDVH', 356, 5)
    results = history.sum(axis=2) """
    
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