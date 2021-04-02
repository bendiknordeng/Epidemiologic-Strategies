import numpy as np
from collections import namedtuple
from covid import utils
import os
from random import uniform, randint

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
        self.paths = utils.create_named_tuple('filepaths.txt')
        param = namedtuple('param', 'R0 DE DI hospitalisation_rate hospital_duration eff OD population')
        self.par = param(R0=R0,
                        DE=DE,
                        DI=DI,
                        hospitalisation_rate=hospitalisation_rate,
                        eff=eff,
                        hospital_duration=hospital_duration,
                        OD=OD,
                        population=population)

    def scale_flow(self, alpha):
        """ Scales flow of individuals between regions

        Parameters
            alpha: array of scalers that adjust flows for a given compartment and region
        Returns
            realflow, scaled flow
        """
        realflow = self.par.OD.copy() 
        realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]  # Normalize flow
        realflow = alpha * realflow 
        return realflow

    def add_hidden_cases(self, s_vec, i_vec, new_i):
        """ Adds cases to the infection compartment, to represent hidden cases

        Parameters
            s_vec: array of susceptible in each region
            i_vec: array of infected in each region
            new_i: array of new cases of infected individuals
        Returns
            new_i, an array of new cases including hidden cases
        """
        share = 0.1 # maximum number of hidden infections
        i_vec = i_vec.reshape(-1) # ensure correct shape
        s_vec = s_vec.reshape(-1) # ensure correct shape
        new_i = new_i.reshape(-1) # ensure correct shape
        for i in range(len(i_vec)):
            if i_vec[i] < 0.5:
                new_infections = uniform(0, 0.01) # introduce infection to region with little infections
            else:
                new_infections = randint(0, min(int(i_vec[i]*share), 1))
            if s_vec[i] > new_infections:
                new_i[i] += new_infections
        return new_i

    def simulate(self, state, decision, decision_period, information, hidden_cases=True, write_to_csv=False, write_weekly=True):
        """  Simulates the development of an epidemic as modelled by current parameters

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
        
        s_vec = state.S
        e_vec = state.E
        i_vec = state.I
        r_vec = state.R
        h_vec = state.H
        v_vec = state.V
        
        result = np.zeros((decision_period, k))
        result[0,:] = [s_vec.sum(), e_vec.sum(), i_vec.sum(), r_vec.sum(), 0, v_vec.sum()]
        
        # Realflows for different comself.partments 
        alpha_s, alpha_e, alpha_i, alpha_r = information['alphas']
        realflow_s = self.scale_flow(alpha_s)
        realflow_e = self.scale_flow(alpha_e)
        realflow_i = self.scale_flow(alpha_i)
        realflow_r = self.scale_flow(alpha_r)
        
        history = np.zeros((decision_period, k, n))
        history[0,0,:] = s_vec
        history[0,1,:] = e_vec
        history[0,2,:] = i_vec
        history[0,3,:] = r_vec
        history[0,4,:] = h_vec
        history[0,5,:] = v_vec

        total_new_infected = np.zeros(decision_period+1)
        
        # run simulation
        for i in range(0, decision_period - 1):
            realOD_s = realflow_s[i % r]
            realOD_e = realflow_e[i % r]
            realOD_i = realflow_i[i % r]
            realOD_r = realflow_r[i % r]
            realOD_v = decision[i % r]

            new_e = s_vec * i_vec / self.par.population.population.to_numpy(dtype='float64') * (self.par.R0 / self.par.DI)
            new_i = e_vec / self.par.DE
            if hidden_cases and (i % (decision_period/7) == 0): 
                new_i = self.add_hidden_cases(s_vec, i_vec, new_i)
            new_r = i_vec / self.par.DI
            new_v = realOD_v * self.par.eff
            
            s_vec -= new_e
            s_vec = (s_vec 
                + np.matmul(s_vec.reshape(1,n), realOD_s)
                - s_vec * realOD_s.sum(axis=1)
                - new_v)

            e_vec = e_vec + new_e - new_i
            e_vec = (e_vec 
                + np.matmul(e_vec.reshape(1,n), realOD_e)
                - e_vec * realOD_e.sum(axis=1))
                    
            i_vec = i_vec + new_i - new_r
            i_vec = (i_vec 
                + np.matmul(i_vec.reshape(1,n), realOD_i)
                - i_vec * realOD_i.sum(axis=1))
                    
            r_vec += new_r
            r_vec = (r_vec 
                + np.matmul(r_vec.reshape(1,n), realOD_r)
                - r_vec * realOD_r.sum(axis=1)
                + new_v)

            v_vec += new_v   
            
            result[i + 1,:] = [s_vec.sum(), e_vec.sum(), i_vec.sum(), r_vec.sum(),  h_vec.sum(), v_vec.sum()]
            total_new_infected[i + 1] = new_i.sum()
            result[i + 1, 4] = total_new_infected[max(0, i - self.par.hospital_duration) : i].sum() * self.par.hospitalisation_rate
            
            history[i + 1,0,:] = s_vec
            history[i + 1,1,:] = e_vec
            history[i + 1,2,:] = i_vec
            history[i + 1,3,:] = r_vec
            history[i + 1,5,:] = v_vec

        if write_to_csv:
            utils.write_history(write_weekly,
                                history, 
                                self.par.population, 
                                state.time_step, 
                                self.paths.results_weekly, 
                                self.paths.results_history)
            
        return result, total_new_infected.sum(), history