from covid.simulation import *
from covid.utils import *
from vaccine_allocation_model.MDP import MarkovDecisionProcess
from collections import namedtuple


if __name__ == '__main__':
    
    # read filepaths 
    paths = create_named_tuple('filepaths.txt')
    
    # read in data from filepaths 
    config = create_named_tuple(paths.config)
    OD_matrices = read_pickle(paths.od)
    vaccine_supply = read_pickle(paths.municipalities_v)
    population = create_population(paths.muncipalities_names, paths.muncipalities_pop)
    seir = initialize_seir(OD_matrices, population, config)

    # # run simulation
    horizon = 50 # number of weeks
    mdp = MarkovDecisionProcess(OD_matrices, population, seir, vaccine_supply, horizon, decision_period=28, policy="population_based")
    path = mdp.run()

    # load necessary data for geospatial plot
    df = pd.read_csv(paths.results_history)
    history = transform_df_to_history(df, 'SEIRHV')
    results = history.sum(axis=2)
    kommuner_geometry = create_geopandas(True, population, paths.municipalities_geo_pkl, paths.municipalities_geo_geojson)

    # accumulated SEIR development plot
    seir_plot(results)
    
    # geospatial plot
    plot_simulation(history[::4,:,:], population, results[::4,4], kommuner_geometry, paths.municipality_plots)

    # generate gif 
    create_gif(paths.municipality_gif,paths.municipality_plots)

    
    
