from covid import plot
from covid import utils
import numpy as np
import pandas as pd
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from covid.seir import SEIR
from covid.seaiqr import SEAIQR

if __name__ == '__main__':
    # read filepaths 
    paths = utils.create_named_tuple('filepaths.txt')
    
    # read in data from filepaths 
    config = utils.create_named_tuple(paths.config)
    population = utils.create_population(paths.muncipalities_names, paths.muncipalities_pop)
    OD_matrices = utils.generate_ssb_od_matrix(28, population, paths.municipalities_commute)
    
    #vaccine_supply = read_pickle(paths.municipalities_v)
    vaccine_supply = np.ones((28,356))
    seaiqrdvh = SEAIQR(OD_matrices,
                population,
                R0=config.R0,
                DE= config.DE* config.periods_per_day,
                DI= config.DI* config.periods_per_day,
                hospitalisation_rate=config.hospitalisation_rate,
                hospital_duration=config.hospital_duration*config.periods_per_day,
                efficacy=config.efficacy,
                proportion_symptomatic_infections=config.proportion_symptomatic_infections,
                latent_period=config.latent_period*config.periods_per_day, 
                recovery_period=config.recovery_period*config.periods_per_day,
                pre_isolation_infection_period=config.pre_isolation_infection_period*config.periods_per_day, 
                post_isolation_recovery_period=config.post_isolation_recovery_period*config.periods_per_day, 
                fatality_rate_symptomatic=config.fatality_rate_symptomatic,
                immunity_duration=config.immunity_duration*config.periods_per_day 
                )


    # run simulation
    horizon = 20 # number of weeks
    mdp = MarkovDecisionProcess(OD_matrices, population, seaiqrdvh, vaccine_supply, horizon, decision_period=28, policy="population_based")
    path = mdp.run()

    # load necessary data for SEIR development plot
    df = pd.read_csv(paths.results_history)
    history = utils.transform_df_to_history(df, 'SEAIQRDVH')
    results = history.sum(axis=2)
    
    # accumulated SEIR development plot
    plot.seir_plot(results)

    # load necessary data for geospatial plot 
    kommuner_geometry = plot.create_geopandas(True, population, paths.municipalities_geo_pkl, paths.municipalities_geo_geojson)

    # geospatial plot of simulation
    plot.plot_simulation(history[::4,:,:], population, results[::4,4], kommuner_geometry, paths.municipality_plots)
    plot.create_gif(paths.municipality_gif,paths.municipality_plots)

    # geospatial plot of of historical covid cases in Norway
    # history = utils.transform_historical_df_to_history(pd.read_csv(paths.municipalities_hist_data))
    # plot.plot_historical_infected(history[::1,:,:], population, kommuner_geometry, paths.municipality_hist_plots)
    # plot.create_gif(paths.municipality_hist_gif,paths.municipality_hist_plots)

    







    
    
