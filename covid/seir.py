import numpy as np
from collections import namedtuple
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class SEIR:
    def __init__(self, OD, pop, R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, hospitalisation_rate=0.1, eff=0.95, hospital_duration=15*12, time_delta=6):
        """ 
        Parameters
        - self.par: parameters {
                    R0: Basic reproduction number (e.g 2.4)
                    DE: Incubation period (e.g 5.6 * 12)        # Needs to multiply by 12 to get one day effects
                    DI: Infectious period (e.g 5.2 * 12)
                    I0: Array with the distribution of infected people at time t=0
                    hospitalisation_rate: Percentage of people that will be hospitalized (e.g 0.1)
                    hospital_duration: Length of hospitalization (e.g 15*12) }
                    eff: vaccine efficacy (e.g 0.95)
                    time_delta: time increment.
         """
        param = namedtuple('param', 'R0 DE DI hospitalisation_rate hospital_duration eff time_delta OD pop')
        self.par = param(R0=R0,
                        DE=DE,
                        DI=DI,
                        hospitalisation_rate=hospitalisation_rate,
                        eff=eff,
                        hospital_duration=hospital_duration,
                        time_delta=time_delta,
                        OD=OD,
                        pop=pop)

    # I0 is the distribution of infected people at time t=0, if None then randomly choose inf number of people

    # flow is a 3D matrix of dimensions r x n x n (i.e., 84 x 549 x 549),
    # flow[t mod r] is the desired OD matrix at time t.

    def scale_flow(self, flow, alpha):
        """scales realflow
        Parameters
            flow: 3D array with flows
            alpha: array of scalers that adjust flows for a given compartment and region
        Returns
            Scaled realflow
        """
        realflow = self.par.OD.copy() 
        realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]  # Normalize flow
        realflow = alpha * realflow 
        return realflow

    def simulate(self, state, decision, days, information):
        """  
        Parameters:
            state: State object with values for each compartment
            decision: vaccine allocation for each period for each region (24/time_delta * days, 356)
            comp_values: dict of values for each compartment (S, E, I, R)
            information: dict of exogenous information for each region (time_delta * days, 356, 356)
        Returns:
            res: accumulated SEIR values for the whole country
            history: SEIRHV for each region for each time step
        """
        iterations = 24/self.par.time_delta * days
        k = 6 # Num of compartments
        r = self.par.OD.shape[0]
        n = self.par.OD.shape[1]
        N = self.par.pop.sum()
        
        Svec = state.S
        Evec = state.E
        Ivec = state.I
        Rvec = state.R
        Hvec = state.H
        Vvec = state.V
        
        res = np.zeros((iterations, k))
        res[0,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(), 0, Vvec.sum()]
        
        # Realflows for different comself.partments 
        alpha_s, alpha_e, alpha_i, alpha_r = information['alpha']
        realflow_s = self.scale_flow(self.par.OD.copy(), alpha_s)
        realflow_e = self.scale_flow(self.par.OD.copy(), alpha_e)
        realflow_i = self.scale_flow(self.par.OD.copy(), alpha_i)
        realflow_r = self.scale_flow(self.par.OD.copy(), alpha_r)
        
        history = np.zeros((iterations, k, n))
        history[0,0,:] = Svec
        history[0,1,:] = Evec
        history[0,2,:] = Ivec
        history[0,3,:] = Rvec
        history[0,4,:] = Hvec
        history[0,5,:] = Vvec

        eachIter = np.zeros(iterations + 1)
        
        # run simulation
        for iter in range(0, iterations - 1):
            realOD_s = realflow_s[iter % r]
            realOD_e = realflow_e[iter % r]
            realOD_i = realflow_i[iter % r]
            realOD_r = realflow_r[iter % r]
            realOD_v = decision[iter % r]

            newE = Svec * Ivec / self.par.pop * (self.par.R0 / self.par.DI)
            newI = Evec / self.par.DE
            newR = Ivec / self.par.DI
            newV = realOD_v * self.par.eff
            
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
            
            res[iter + 1,:] = [Svec.sum(), Evec.sum(), Ivec.sum(), Rvec.sum(),  Hvec.sum(), Vvec.sum()]
            eachIter[iter + 1] = newI.sum()
            res[iter + 1, 4] = eachIter[max(0, iter - self.par.hospital_duration) : iter].sum() * self.par.hospitalisation_rate
            
            history[iter + 1,0,:] = Svec
            history[iter + 1,1,:] = Evec
            history[iter + 1,2,:] = Ivec
            history[iter + 1,3,:] = Rvec
            history[iter + 1,5,:] = Vvec

        return res, history