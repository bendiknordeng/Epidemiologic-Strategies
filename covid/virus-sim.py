
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import namedtuple
import pickle
import pandas as pd
import matplotlib.pyplot as plt

Param = namedtuple('Param', 'R0 DE DI I0 HospitalisationRate HospitalIters eff')
# I0 is the distribution of infected people at time t=0, if None then randomly choose inf number of people

# flow is a 3D matrix of dimensions r x n x n (i.e., 84 x 549 x 549),
# flow[t mod r] is the desired OD matrix at time t.

def scale_flow(flow, alpha):
    """scales realflow
    Parameters:
        flow: 3D array with flows
        alpha: array of scalers that adjust flows for a given compartment and region
    Return:
        Scaled realflow
    """
    realflow = flow.copy() # copy!
    realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]  # Normalize the flow
    realflow = alpha * realflow 
    return realflow


def seir(par, distr, flow, alphas, iterations, inf, vacc):
    """ Simulates an epidemic
    Parameters:
        - par: parameters {
                R0: Basic reproduction number (e.g 2.4)
                DE: Incubation period (e.g 5.6 * 12)        # Needs to multiply by 12 to get one day effects
                DI: Infectious period (e.g 5.2 * 12)
                I0: Array with the distribution of infected people at time t=0
                HospitalisationRate: Percentage of people that will be hospitalized (e.g 0.1)
                HospitalIters: Length of hospitalization (e.g 15*12) }
                eff: vaccine efficacy (e.g 0.95)
        - distr: population distribution
        - flow: OD matrices with dimensions r x n x n (i.e., 84 x 549 x 549).  flow[t mod r] is the desired OD matrix at time t. Use mod in order to only need one week of OD- matrices. 
        - alphas: [alpha_s, alpha_e, alpha_i, alpha_r] strength of lock down measures/movement restriction. value of 1 - normal flow, 0 - no flow 
        - iterations: number of simulations/ duration of simulation
        - inf: number of infections
        - vacc: 
    Returns: 
        - res: matrix of shape (#iterations, #compartments" + 1(hospitalized people))
        - history: matrix with the number of subjects in each compartment [sim_it, compartment_id, num_cells]
    """
    
    k = 6 # Num of compartments
    r = flow.shape[0]
    n = flow.shape[1]
    N = distr[0].sum() # total population, we assume that N = sum(flow)
    
    Svec = distr[0].copy()
    Evec = np.zeros(n)
    Ivec = np.zeros(n)
    Rvec = np.zeros(n)
    Vvec = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf):
            loc = np.random.randint(n)
            if (Svec[loc] > initial[loc]):
                initial[loc] += 1.0

    else:
        initial = par.I0
    assert ((Svec < initial).sum() == 0)
    
    Svec -= initial
    Ivec += initial
    
    res = np.zeros((iterations, k))
    res[0,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0, Vvec.sum()]
    
    # Realflows for different compartments 
    alpha_s, alpha_e, alpha_i, alpha_r = alphas
    realflow_s = scale_flow(flow.copy(), alpha_s)
    realflow_e = scale_flow(flow.copy(), alpha_e)
    realflow_i = scale_flow(flow.copy(), alpha_i)
    realflow_r = scale_flow(flow.copy(), alpha_r)
    
    history = np.zeros((iterations, k, n))
    history[0,0,:] = Svec
    history[0,1,:] = Evec
    history[0,2,:] = Ivec
    history[0,3,:] = Rvec
    history[0,5,:] = Vvec

    
    eachIter = np.zeros(iterations + 1)
    
    # run simulation
    for iter in range(0, iterations - 1):
        realOD_s = realflow_s[iter % r]
        realOD_e = realflow_e[iter % r]
        realOD_i = realflow_i[iter % r]
        realOD_r = realflow_r[iter % r]
        
        v = vacc[iter % r] + 1
        v = np.minimum(v, Svec)[0] # Ensure that no people are vaccinated if they are not suceptible
    
        d = distr[iter % r] + 1  # At least one person in each cell.
        
        if ((d>N+1).any()): #assertion!
            print("Population does not stay constant!")
            return res, history
        print(par.R0)    
        newE = Svec * Ivec / d * (par.R0 / par.DI)
        newI = Evec / par.DE
        newR = Ivec / par.DI
        newV = v * par.eff
        
        Svec -= newE
        Svec = (Svec 
               + np.matmul(Svec.reshape(1,n), realOD_s)
               - Svec * realOD_s.sum(axis=1)
               - newV
                )
        Evec = Evec + newE - newI
        Evec = (Evec 
               + np.matmul(Evec.reshape(1,n), realOD_e)
               - Evec * realOD_e.sum(axis=1)
                )
                
        Ivec = Ivec + newI - newR
        Ivec = (Ivec 
               + np.matmul(Ivec.reshape(1,n), realOD_i)
               - Ivec * realOD_i.sum(axis=1)
                )
                
        Rvec += newR
        Rvec = (Rvec 
               + np.matmul(Rvec.reshape(1,n), realOD_r)
               - Rvec * realOD_r.sum(axis=1)
               + newV
                )

        Vvec += newV   
        
        res[iter + 1,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0, Vvec.sum()]
        eachIter[iter + 1] = newI.sum()
        res[iter + 1, 4] = eachIter[max(0, iter - par.HospitalIters) : iter].sum() * par.HospitalisationRate
        
        history[iter + 1,0,:] = Svec
        history[iter + 1,1,:] = Evec
        history[iter + 1,2,:] = Ivec
        history[iter + 1,3,:] = Rvec
        history[iter + 1,5,:] = Vvec


    return res, history

def seir_plot_one_cell(res, cellid):
    """ Plots SIR for a single cell
    Parameters:
    res: [3D array, compartment_id]
    """
    plt.plot(res[::12, 0, cellid], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1, cellid], color='g', label='E')
    plt.plot(res[::12, 2, cellid], color='b', label='I')
    plt.plot(res[::12, 3, cellid], color='y', label='R')
    plt.plot(res[::12, 4, cellid], color='c', label='H')
    plt.plot(res[::12, 5, cellid], color='m', label='V')
    plt.legend()
    plt.show()

def seir_plot(res):
    """ Plots the epidemiological curves
    Parameters:
    res: [3D array, compartment_id]
    """
    plt.plot(res[::12, 0], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.plot(res[::12, 5], color='m', label='V')
    plt.legend()
    plt.show()

"""
def main():

    # load OD matrices
    pkl_file = open('covid/data/data_counties/od_counties.pkl', 'rb') 
    OD_matrices = pickle.load(pkl_file)
    pkl_file.close()

    # load vaccine schedule
    pkl_file = open('covid/data/data_counties/vaccines_counties.pkl', 'rb') 
    vacc = pickle.load(pkl_file)
    pkl_file.close()
    print(vacc.shape)

    # create population 
    df_befolkningstall_fylker = pd.read_csv("covid/data/data_counties/Folkemengde_fylker.csv", delimiter=";", skiprows=1)
    befolkningsarray= df_befolkningstall_fylker['Befolkning per 1.1. (personer) 2020'].to_numpy(dtype='float64')
    pop = np.asarray([befolkningsarray for _ in range(84)])

    # Define simulation parameters
    r = OD_matrices.shape[0]  # Simulation period (e.g 84)
    n = pop.shape[1]          # Number of counties (e.g 11)
    N = sum(befolkningsarray) # Total population (e.g 5367580)
    initialInd = [2]          # Initial index of counties infected
    initial = np.zeros(n)
    initial[initialInd] = 50  # Number of infected people in each of the initial counties infected

    model = Param(R0=2.4, DE=5.6*12, DI=5.2*12, I0=initial, HospitalisationRate=0.1, HospitalIters=15*12, eff=0.9) # multiply by 12 as one day consists of 12 2-hours periods 

    alpha = np.ones(OD_matrices.shape)  # One == no quarantene influence. Multiplied by real flow.
    iterations = 3000                    # Number of simulations
    res = {}                            # Dictionary with results for different cases 
    inf = 50                            # Number of random infections
    res['baseline'] = seir(model, pop, OD_matrices, alpha, iterations, inf, vacc)
    seir_plot(res["baseline"][0])
    


if __name__ == '__main__':
    main()
"""