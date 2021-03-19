from simulation import *
from collections import namedtuple
from utils import read_config

if __name__ == '__main__':
    config = read_config()
    """
    OD_matrices = load_od_matrices()
    pop, befolkning = create_population()
    seir = initialize_seir(config, OD_matrices.shape[0], pop.shape[1], sum(pop[0]), 50)
    vacc = load_vaccination_programme(OD_matrices.shape[0], pop.shape[1])
    res = {}                            # Dictionary with results for different cases 
    res['baseline'] = simulate(seir, pop, OD_matrices, vacc)
    kommuner_geometry = create_geopandas(geopandas_from_pkl=True)
    # Print hospitalized information
    print("Max number of hospitalised people: ", int(res["baseline"][0][:,4].max()))
    print("Day with max hospitalised people: ", int(res["baseline"][0][:,4].argmax()/12)) # Divide by
    seir_plot(res["baseline"][0])
    # declare baseline array storing the dynamics of the compartments 
    baseline = res['baseline'][1][::12, :, :]
    print(baseline.shape)

    # declare hopsitalisation array storing the dynamics of the hospitalised 
    hosp = res['baseline'][0][::12, 4]
    print(hosp.shape)

    max_exp_ind, max_exp_val = find_max_exposed(baseline, befolkning)
    plot_simulation(kommuner_geometry)
    create_gif()
    """