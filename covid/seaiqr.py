import numpy as np
from collections import namedtuple
from covid import utils
import os

class SEAIQR:
    def __init__(self, OD, population, R0=2.4, DE= 5.6*4, DI= 5.2*4, hospitalisation_rate=0.1, hospital_duration=15*4,
    efficacy=0.95,  proportion_symptomatic_infections=0.8, latent_period=5.1*4, recovery_period=21*4,
    pre_isolation_infection_period=4.6*4, post_isolation_recovery_period=16.4*4, fatality_rate_symptomatic=0.01*4,
    immunity_duration=365*4
    ):
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
                    efficacy: vaccine efficacy (e.g 0.95)
                    proportion_symptomatic_infections: Proportion of symptomatic infections(e.g 0.8)
                    latent_period: (e.g 5.1)
                    recovery_period:   (e.g 21)
                    pre_isolation_infection_period: Pre-isolation infection period (e.g 4.6)
                    post_isolation_recovery_period: (e.g 16.4)
                    fatality_rate_symptomatic: (e.g 0.01)
                    immunity_duration: (e.g 365)
         """
        self.paths = utils.create_named_tuple('filepaths.txt')
        param = namedtuple('param', 'OD population R0 DE DI hospitalisation_rate hospital_duration efficacy proportion_symptomatic_infections latent_period recovery_period pre_isolation_infection_period post_isolation_recovery_period fatality_rate_symptomatic immunity_duration')
        self.par = param(
                        OD=OD,
                        population=population, 
                        R0=R0, 
                        DE=DE, 
                        DI=DI, 
                        hospitalisation_rate=hospitalisation_rate, 
                        hospital_duration=hospital_duration,
                        efficacy=efficacy,  
                        proportion_symptomatic_infections=proportion_symptomatic_infections, 
                        latent_period=latent_period, 
                        recovery_period=recovery_period,
                        pre_isolation_infection_period=pre_isolation_infection_period, 
                        post_isolation_recovery_period=post_isolation_recovery_period, 
                        fatality_rate_symptomatic=fatality_rate_symptomatic,
                        immunity_duration=immunity_duration
                        )

    def scale_flow(self, alpha):
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
        # Meta-parameters
        compartments = 'SEAIQRDVH'
        k = len(compartments)
        r = self.par.OD.shape[0]
        n = self.par.OD.shape[1]
        
        S_vec = state.S
        E_vec = state.E
        A_vec = state.A
        I_vec = state.I
        Q_vec = state.Q
        R_vec = state.R
        D_vec = state.D
        V_vec = state.V
        H_vec = state.H
        
        result = np.zeros((decision_period, k))
        result[0,:] = [S_vec.sum(), E_vec.sum(), A_vec.sum(), I_vec.sum(), Q_vec.sum(), R_vec.sum(), D_vec.sum(), V_vec.sum(), 0]
        
        # Realflows for different comself.partments 
        alpha_s, alpha_e, alpha_i, alpha_r = information['alphas'] # They currently have the same values
        realflow_s = self.scale_flow(alpha_s)
        realflow_e = self.scale_flow(alpha_e)
        realflow_i = self.scale_flow(alpha_i)
        realflow_r = self.scale_flow(alpha_r)
        
        history = np.zeros((decision_period, k, n))
        history[0,0,:] = S_vec
        history[0,1,:] = E_vec
        history[0,2,:] = A_vec
        history[0,3,:] = I_vec
        history[0,4,:] = Q_vec
        history[0,5,:] = R_vec
        history[0,6,:] = D_vec
        history[0,7,:] = V_vec
        history[0,8,:] = H_vec


        total_new_infected = np.zeros(decision_period+1)
        
        # run simulation
        for i in range(0, decision_period - 1):
            # Fins the flow between regions for each compartment 
            realOD_s = realflow_s[i % r]
            realOD_e = realflow_e[i % r]
            realOD_a = realflow_e[i % r] # Obs. Change this later using realflow_a
            realOD_i = realflow_i[i % r]
            realOD_q = realflow_e[i % r] # Obs. Change this later using realflow_q
            realOD_r = realflow_r[i % r]
            
            # Finds the decision - number of vaccines to be allocated to each region for a specific time period
            v = decision[i % r]

            # Finds values for each arraow in epidemic model 
            newS = R_vec / self.par.immunity_duration
            newE = S_vec * (A_vec + I_vec) / self.par.population.population.to_numpy(dtype='float64') * (self.par.R0 / self.par.DI)  # Need to change this to force of infection 
            newA = (1 - self.par.proportion_symptomatic_infections) * E_vec / self.par.latent_period
            newI = self.par.proportion_symptomatic_infections *  E_vec / self.par.latent_period
            newQ = I_vec /  self.par.pre_isolation_infection_period  
            newR_fromA = A_vec / self.par.recovery_period
            newR_fromQ = Q_vec * (1- self.par.fatality_rate_symptomatic) / self.par.recovery_period 
            newR_fromV = V_vec/self.par.latent_period
            newD = Q_vec * self.par.fatality_rate_symptomatic / self.par.recovery_period
            newV = v * self.par.efficacy

            # Calculate new values for each compartment
            S_vec = S_vec + newS - newV 
            S_vec = (S_vec 
                + np.matmul(S_vec.reshape(1,n), realOD_s)
                - S_vec * realOD_s.sum(axis=1))
            E_vec = E_vec + newE - newI - newA
            E_vec = (E_vec 
                + np.matmul(E_vec.reshape(1,n), realOD_e)
                - E_vec * realOD_e.sum(axis=1))
            A_vec = A_vec + newA - newR_fromA
            A_vec = (A_vec 
                + np.matmul(A_vec.reshape(1,n), realOD_a)
                - A_vec * realOD_a.sum(axis=1))
            I_vec = I_vec + newI - newQ
            I_vec = (I_vec 
                + np.matmul(I_vec.reshape(1,n), realOD_i)
                - I_vec * realOD_i.sum(axis=1))
            Q_vec = Q_vec + newQ - newR_fromQ - newD
            Q_vec = (Q_vec 
                + np.matmul(Q_vec.reshape(1,n), realOD_q)
                - Q_vec * realOD_q.sum(axis=1))
            R_vec = R_vec + newR_fromQ + newR_fromA + newR_fromV
            R_vec = (R_vec 
                + np.matmul(R_vec.reshape(1,n), realOD_r)
                - R_vec * realOD_r.sum(axis=1))
            D_vec = D_vec + newD
            V_vec = V_vec + newV - newR_fromV

            # Add the accumulated numbers to results
            result[i + 1,:] = [S_vec.sum(), E_vec.sum(), A_vec.sum(), I_vec.sum(), Q_vec.sum(), R_vec.sum(),  D_vec.sum(), H_vec.sum(), V_vec.sum()]
            
            # Add number of hospitalized 
            total_new_infected[i + 1] = newI.sum()
            result[i + 1, 8] = total_new_infected[max(0, i - self.par.hospital_duration) : i].sum() * self.par.hospitalisation_rate
            
            history[i + 1,0,:] = S_vec
            history[i + 1,1,:] = E_vec
            history[i + 1,2,:] = A_vec
            history[i + 1,3,:] = I_vec
            history[i + 1,4,:] = Q_vec
            history[i + 1,5,:] = R_vec
            history[i + 1,6,:] = D_vec
            history[i + 1,7,:] = V_vec
            history[i + 1,8,:] = H_vec

        if write_to_csv:
            if write_weekly:
                latest_df = utils.transform_history_to_df(state.time_step, np.expand_dims(history[-1], axis=0), self.par.population, compartments)
                if os.path.exists(self.paths.results_weekly):
                    if state.time_step == 0: # block to remove old csv file if new run is executed 
                        os.remove(self.paths.results_weekly)
                        latest_df.to_csv(self.paths.results_weekly)
                    else:
                        latest_df.to_csv(self.paths.results_weekly, mode='a', header=False)
                else:
                    latest_df.to_csv(self.paths.results_weekly)
            else:
                history_df = utils.transform_history_to_df(state.time_step, history, self.par.population, compartments)
                if os.path.exists(self.paths.results_history):
                    if state.time_step == 0: # block to remove old csv file if new run is executed
                        os.remove(self.paths.results_history)
                        history_df.to_csv(self.paths.results_history)
                    else:
                        history_df.to_csv(self.paths.results_history, mode='a', header=False)
                else:
                    history_df.to_csv(self.paths.results_history)
        
        return result, total_new_infected.sum(), history