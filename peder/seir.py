import numpy as np
import matplotlib.pyplot as plt

def seir(par, distr, flow, alpha, iterations, inf):
    """ Simulates an epidemic
    Parameters:
        - par: parameters {
                R0: Basic reproduction number (e.g 2.4)
                DE: Incubation period (e.g 5.6 * 12)        # Needs to multiply by 12 to get one day effects
                DI: Infectious period (e.g 5.2 * 12)
                I0: In(eitial infectiouns .g initial)
                HospitalisationRate: Percentage of people that will be hospitalized (e.g 0.1)
                HospitalIters: Length of hospitalization (e.g 15*12) }
        - distr: population distribution
        - flow: OD matrices with dimensions r x n x n (i.e., 84 x 549 x 549).  flow[t mod r] is the desired OD matrix at time t. Use mod in order to only need one week of OD- matrices. 
        - alpha: strength of lock down measures/movement restriction. value of 1 - normal flow, 0 - no flow 
        - iterations: number of simulations/ duration of simulation
        - inf: number of infections
    Returns: 
        - res: matrix of shape (#iterations, #compartments" + 1(hospitalized people))
        - history: matrix with the number of subjects in each compartment [sim_it, compartment_id, num_cells]
    """
    
    r = flow.shape[0]  # Number of OD matrices - Ex. 84  
    n = flow.shape[1]  # Number of regions - Ex. 549
    N = distr[0].sum() # Total population. Assumes that N = sum(flow) 

    # Initialize compartments
    Svec = distr[0].copy()
    Evec = np.zeros(n)
    Ivec = np.zeros(n)
    Rvec = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf):
            loc = np.random.randint(n)
            if (Svec[loc] > initial[loc]):
                initial[loc] += 1.0      
    else:
        initial = par.I0
    assert ((Svec < initial).sum() == 0) # Make sure that number of suceptible > infected for every region
    
    Svec -= initial
    Ivec += initial
    
    res = np.zeros((iterations, 5))
    res[0,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
    
    # Normalise the flows and then multiply them by the alpha values. 
    # Alpha is just a conceptual measure that reduce the mobility flow
    realflow = flow.copy()
    realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]    
    realflow = alpha * realflow 
    
    history = np.zeros((iterations, 5, n)) # 5 beacuse of number of compartments
    history[0,0,:] = Svec
    history[0,1,:] = Evec
    history[0,2,:] = Ivec
    history[0,3,:] = Rvec
    
    eachIter = np.zeros(iterations + 1)
    
    # run simulation
    for iter in range(0, iterations - 1):
        realOD = realflow[iter % r]  # Stays within one week using modelo
        
        d = distr[iter % r] + 1  # At least one person in each cell. To avoid normalization problems. 

        newE = Svec * Ivec / d * (par.R0 / par.DI)
        newI = Evec / par.DE
        newR = Ivec / par.DI
        
        Svec -= newE
        Svec = (Svec 
               + np.matmul(Svec.reshape(1,n), realOD)
               - Svec * realOD.sum(axis=1)
                )
        Evec = Evec + newE - newI
        Evec = (Evec 
               + np.matmul(Evec.reshape(1,n), realOD)
               - Evec * realOD.sum(axis=1)
                )
                
        Ivec = Ivec + newI - newR
        Ivec = (Ivec 
               + np.matmul(Ivec.reshape(1,n), realOD)
               - Ivec * realOD.sum(axis=1)
                )
                
        Rvec += newR
        Rvec = (Rvec 
               + np.matmul(Rvec.reshape(1,n), realOD)
               - Rvec * realOD.sum(axis=1)
                )
        
        # Stores the historical S, E, I, C in each cell
        res[iter + 1,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0]
        eachIter[iter + 1] = newI.sum()
        res[iter + 1, 4] = eachIter[max(0, iter - par.HospitalIters) : iter].sum() * par.HospitalisationRate
        
        # Stores the historical S, E, I, C in each cell 
        history[iter + 1,0,:] = Svec
        history[iter + 1,1,:] = Evec
        history[iter + 1,2,:] = Ivec
        history[iter + 1,3,:] = Rvec    
    return res, history


def seir_plot(res):
    """ Plots the epidemiological curves
    Parameters:
        res: [3D matrix, compartment_id]
    """
    plt.plot(res[::12, 0], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1], color='g', label='E')
    plt.plot(res[::12, 2], color='b', label='I')
    plt.plot(res[::12, 3], color='y', label='R')
    plt.plot(res[::12, 4], color='c', label='H')
    plt.legend()
    plt.show()


def seir_plot_one_cell(res, cellid):
    """ Plots SIR for a single cell
    Parameters:
        res: [3D matrix, compartment_id]
    """
    plt.plot(res[::12, 0, cellid], color='r', label='S') # Take every 12 value to get steps per day (beacause of 2-hours intervals) 
    plt.plot(res[::12, 1, cellid], color='g', label='E')
    plt.plot(res[::12, 2, cellid], color='b', label='I')
    plt.plot(res[::12, 3, cellid], color='y', label='R')
    plt.plot(res[::12, 4, cellid], color='c', label='H')
    plt.legend()
    plt.show()



