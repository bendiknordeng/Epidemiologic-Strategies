import numpy as np
from collections import namedtuple
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from covid import utils
import os
from covid.utils import *

class SEIR:
    def __init__(self, OD, population, R0=2.4, DE= 5.6 * 12, DI= 5.2 * 12, hospitalisation_rate=0.1, eff=0.95, hospital_duration=15*12):
        """ 
        Parameters
        - self.par: parameters {
                    OD: Origin-Destination matrix
                    population: pd.DataFrame with columns region_id, region_name, population (quantity)
                    R0: Basic reproduction number (e.g 2.4)
                    DE: Incubation period (e.g 5.6 * 12)        # Needs to multiply by 12 to get one day effects
                    DI: Infectious period (e.g 5.2 * 12)
                    hospitalisation_rate: Percentage of people that will be hospitalized (e.g 0.1)
                    hospital_duration: Length of hospitalization (e.g 15*12) }
                    eff: vaccine efficacy (e.g 0.95)
         """
        self.paths = create_named_tuple('filepaths.txt')
        param = namedtuple('param', 'R0 DE DI hospitalisation_rate hospital_duration eff OD population')
        self.par = param(R0=R0,
                        DE=DE,
                        DI=DI,
                        hospitalisation_rate=hospitalisation_rate,
                        eff=eff,
                        hospital_duration=hospital_duration,
                        OD=OD,
                        population=population)

    def scale_flow(self, flow, alpha):
        """ scales realflow
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

    def simulate(self, state, decision, decision_period, information, write_to_csv=False, write_weekly=True):
        """  simulates the development of an epidemic as modelled by current parameters
        Parameters:
            state: State object with values for each compartment
            decision: Vaccine allocation for each period for each region, shape (decision_period, nr_regions)
            decision_period: number of steps the simulation makes
            information: dict of exogenous information for each region, shape (decision_period, nr_regions, nr_regions)
            write_to_csv: Bool, True if history is to be saved as csv
            write_weekly: Bool, True if history is to be sampled on a weekly basis
        Returns:
            res: accumulated SEIR values for all regions as whole (decision_period, )
            total_new_infected.sum(): accumulated infected for the decision_period, float.
            history: SEIRHV for each region for each time step (decision_period,  number_compartments, number_of_regions)
        """
        
        k = 6 # Num of compartments
        r = self.par.OD.shape[0]
        n = self.par.OD.shape[1]
        N = self.par.population.population.to_numpy(dtype='float64').sum()
        
        S_vec = state.S
        E_vec = state.E
        I_vec = state.I
        R_vec = state.R
        H_vec = state.H
        V_vec = state.V
        
        result = np.zeros((decision_period, k))
        result[0,:] = [S_vec.sum(), E_vec.sum(), I_vec.sum(), R_vec.sum(), 0, V_vec.sum()]
        
        # Realflows for different comself.partments 
        alpha_s, alpha_e, alpha_i, alpha_r = information['alphas']
        realflow_s = self.scale_flow(self.par.OD.copy(), alpha_s)
        realflow_e = self.scale_flow(self.par.OD.copy(), alpha_e)
        realflow_i = self.scale_flow(self.par.OD.copy(), alpha_i)
        realflow_r = self.scale_flow(self.par.OD.copy(), alpha_r)
        
        history = np.zeros((decision_period, k, n))
        history[0,0,:] = S_vec
        history[0,1,:] = E_vec
        history[0,2,:] = I_vec
        history[0,3,:] = R_vec
        history[0,4,:] = H_vec
        history[0,5,:] = V_vec

        total_new_infected = np.zeros(decision_period+1)
        
        # run simulation
        for i in range(0, decision_period - 1):
            realOD_s = realflow_s[i % r]
            realOD_e = realflow_e[i % r]
            realOD_i = realflow_i[i % r]
            realOD_r = realflow_r[i % r]
            realOD_v = decision[i % r]

            newE = S_vec * I_vec / self.par.population.population.to_numpy(dtype='float64') * (self.par.R0 / self.par.DI)
            newI = E_vec / self.par.DE
            newR = I_vec / self.par.DI
            newV = realOD_v * self.par.eff
            
            S_vec -= newE
            S_vec = (S_vec 
                + np.matmul(S_vec.reshape(1,n), realOD_s)
                - S_vec * realOD_s.sum(axis=1)
                - newV
                    )
            E_vec = E_vec + newE - newI
            E_vec = (E_vec 
                + np.matmul(E_vec.reshape(1,n), realOD_e)
                - E_vec * realOD_e.sum(axis=1)
                    )
                    
            I_vec = I_vec + newI - newR
            I_vec = (I_vec 
                + np.matmul(I_vec.reshape(1,n), realOD_i)
                - I_vec * realOD_i.sum(axis=1)
                    )
                    
            R_vec += newR
            R_vec = (R_vec 
                + np.matmul(R_vec.reshape(1,n), realOD_r)
                - R_vec * realOD_r.sum(axis=1)
                + newV
                    )

            V_vec += newV   
            
            result[i + 1,:] = [S_vec.sum(), E_vec.sum(), I_vec.sum(), R_vec.sum(),  H_vec.sum(), V_vec.sum()]
            total_new_infected[i + 1] = newI.sum()
            result[i + 1, 4] = total_new_infected[max(0, i - self.par.hospital_duration) : i].sum() * self.par.hospitalisation_rate
            
            history[i + 1,0,:] = S_vec
            history[i + 1,1,:] = E_vec
            history[i + 1,2,:] = I_vec
            history[i + 1,3,:] = R_vec
            history[i + 1,5,:] = V_vec

        if write_to_csv:
            if write_weekly:
                latest_df = utils.transform_history_to_df(state.time_step, np.expand_dims(history[-1], axis=0), self.par.population, "SEIRHV")
                if os.path.exists(self.paths.results_weekly):
                    latest_df.to_csv(self.paths.results_weekly, mode="a", header=False)
                else:
                    latest_df.to_csv(self.paths.results_weekly)
            else:
                history_df = utils.transform_history_to_df(state.time_step, history, self.par.population, "SEIRHV")
                if os.path.exists(self.paths.results_dayly):
                    history_df.to_csv(self.paths.results_dayly, mode="a", header=False)
                else:
                    history_df.to_csv(self.paths.results_dayly)
        
        return result, total_new_infected.sum(), history