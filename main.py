from covid.simulation import *
from collections import namedtuple
from covid.utils import read_config


if __name__ == '__main__':
    # filepaths 
    fpath_config                     = 'configs/baseline.txt'
    fpath_od                         = 'covid/data/data_municipalities/od_municipalities.pkl'
    fpath_muncipalities_names        = 'covid/data/data_municipalities/Kommunenummer_navn_2020.csv'
    fpath_muncipalities_pop          = 'covid/data/data_municipalities/Folkemengde_kommuner.csv'
    fpath_municipalities_v           = 'covid/data/data_municipalities/vaccines_municipalities.pkl'
    fpath_municipalities_geo_pkl     = 'covid/data/data_municipalities/norge_geojson.pkl'
    fpath_municipalities_geo_geojson = 'covid/data/data_municipalities/Basisdata_0000_Norge_25833_Kommuner_GEOJSON.geojson'
    fpath_municipality_gif           = 'covid/gifs/gifs_municipalities/Covid_19_municipalities.gif'
    
    config = read_config(fpath_config)
    
    # read in data from filepaths 
    config = read_config(fpath_config)
    OD_matrices = load_od_matrices(fpath_od)
    pop, befolkning = create_population(fpath_muncipalities_names, fpath_muncipalities_pop)
    seir = initialize_seir(config, pop.shape[1], 50)
    vacc = load_vaccination_programme(OD_matrices.shape[0], pop.shape[1], fpath_municipalities_v)
    kommuner_geometry = create_geopandas(True, befolkning, fpath_municipalities_geo_pkl, fpath_municipalities_geo_geojson)
    
    # simulate seir 
    res = {}   # Dictionary with the results for all cases {}                         
    res['baseline'] = simulate(seir, pop, OD_matrices, vacc)

    # plot seir for all region
    seir_plot(res["baseline"][0])
    
    # declare baseline array storing the dynamics of the compartments 
    baseline = res['baseline'][1][::12, :, :]

    # declare hopsitalisation array storing the dynamics of the hospitalised 
    hosp = res['baseline'][0][::12, 4]

    # geospatial plots 
    #plot_simulation(baseline, befolkning, hosp, kommuner_geometry)

    # generate gif 
    #create_gif(fpath_municipality_gif)