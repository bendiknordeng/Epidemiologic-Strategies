import numpy as np
from collections import namedtuple
from covid import utils
from random import uniform, randint

class SEAIQR:
    def __init__(self, OD, population, contact_matrices, R0=2.4, DE= 5.6*4, DI= 5.2*4, hospitalisation_rate=0.1, hospital_duration=15*4,
    efficacy=0.95,  proportion_symptomatic_infections=0.8, latent_period=5.1*4, recovery_period=21*4,
    pre_isolation_infection_period=4.6*4, post_isolation_recovery_period=16.4*4, fatality_rate_symptomatic=0.01*4,
    immunity_duration=365*4):
        """ 
        Parameters
        - self.par: parameters {
                    OD: Origin-Destination matrix
                    population: pd.DataFrame with columns region_id, region_name, population (quantity)
                    contact_matrices: list of lists with measurement of the intensitity of social contacts among seven age-groups at households, schools, workplaces, and public/community
                    R0: Basic reproduction number (e.g 2.4)
                    DE: Incubation period (e.g 5.6 * 4)        # Needs to multiply by 12 to get one day effects
                    DI: Infectious period (e.g 5.2 * 4)
                    hospitalisation_rate: Percentage of people that will be hospitalized (e.g 0.1)
                    hospital_duration: Length of hospitalization (e.g 15*4) }
                    efficacy: vaccine efficacy (e.g 0.95)
                    proportion_symptomatic_infections: Proportion of symptomatic infections(e.g 0.8)
                    latent_period: Time before vaccine is effective (e.g 5.1*4)
                    recovery_period: Time to recover from receiving the virus to not being  (e.g 21'4)
                    pre_isolation_infection_period: Pre-isolation infection period (e.g 4.6*4)
                    post_isolation_recovery_period: Post-isolation recovery period (e.g 16.4*4)
                    fatality_rate_symptomatic: Fatality rate for people that experience symptoms (e.g 0.01)
                    immunity_duration: Immunity duration of vaccine or after having the disease (e.g 365*4)
         """
        self.paths = utils.create_named_tuple('filepaths.txt')
        param = namedtuple('param', 'OD population contact_matrices R0 DE DI hospitalisation_rate hospital_duration efficacy proportion_symptomatic_infections latent_period recovery_period pre_isolation_infection_period post_isolation_recovery_period fatality_rate_symptomatic immunity_duration')
        self.par = param(
                        OD=OD,
                        population=population,
                        contact_matrices=contact_matrices,
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
    
    # NEED TO UPDATE WITH CONTACT MATRIX!
    def simulate(self, state, decision, decision_period, information, hidden_cases=True, write_to_csv=False, write_weekly=True):
        """  simulates the development of an epidemic as modelled by current parameters
        
        Parameters:
            state: State object with values for each compartment
            decision: Vaccine allocation for each period for each region, shape (decision_period, nr_regions)
            decision_period: number of steps the simulation makes
            information: dict of exogenous information for each region, shape (decision_period, nr_regions, nr_regions)
            write_to_csv: Bool, True if history is to be saved as csv
            write_weekly: Bool, True if history is to be sampled on a weekly basis
            hidden_cases: Bool, True if random hidden infections is to be included in modelling
        Returns:
            res: accumulated SEIR values for all regions as whole (decision_period, )
            total_new_infected.sum(): accumulated infected for the decision_period, float.
            history: SEIRQDHV for each region for each time step (decision_period,  number_compartments, number_of_regions)
        """
        # Meta-parameters
        compartments = 'SEAIQRDVH'
        k = len(compartments)
        r = self.par.OD.shape[0] 
        n = self.par.OD.shape[1]
    
        s_vec, e_vec, a_vec, i_vec, q_vec, r_vec, d_vec, v_vec, h_vec = state.get_compartments_values()
        
        # Extraxt movement information and scale flow
        realflow_s, realflow_e, realflow_a, realflow_i, realflow_q, realflow_r = self.get_realflows(information['alphas'] )
        
        # Initialize output matrices
        history, result = self.initialize_output_matrices(decision_period, k, n, s_vec, e_vec, a_vec, i_vec, q_vec, r_vec, d_vec, v_vec, h_vec)
        total_new_infected = np.zeros(decision_period+1)
        
        # Run simulation
        for i in range(0, decision_period - 1):
            # Finds the flow between regions for each compartment 
            realOD_s, realOD_e, realOD_a, realOD_i, realOD_q, realOD_r = self.get_relevant_flow(r, realflow_s, realflow_e, realflow_a, realflow_i, realflow_q, realflow_r, i)
            
            # Finds the decision - number of vaccines to be allocated to each region for a specific time period
            v = decision[i % r]

            # Calculate values for each arrow in epidemic model 
            new_s = r_vec / self.par.immunity_duration
            new_e = s_vec * (a_vec + i_vec) / self.par.population.population.to_numpy(dtype='float64') * (self.par.R0 / self.par.DI)  # Need to change this to force of infection 
            new_a = (1 - self.par.proportion_symptomatic_infections) * e_vec / self.par.latent_period
            new_i = self.par.proportion_symptomatic_infections *  e_vec / self.par.latent_period

            # Add random infected to new I if it is included in modelling (NEED TO BE ADJUSTED FOR AGE GROUP DIMENSIONS)
            if hidden_cases and (i % (decision_period/7) == 0): 
                new_i = self.add_hidden_cases(s_vec, i_vec, new_i)
                
            new_q = i_vec /  self.par.pre_isolation_infection_period  
            new_r_from_a = a_vec / self.par.recovery_period
            new_r_from_q = q_vec * (1- self.par.fatality_rate_symptomatic) / self.par.recovery_period 
            new_r_from_v = v_vec/self.par.latent_period
            new_d = q_vec * self.par.fatality_rate_symptomatic / self.par.recovery_period
            new_v = v * self.par.efficacy

            # Calculate values for each compartment
            s_vec = s_vec + new_s - new_v - new_e
            s_vec = (s_vec 
                + np.matmul(s_vec.reshape(1,n), realOD_s)
                - s_vec * realOD_s.sum(axis=1))
            e_vec = e_vec + new_e - new_i - new_a
            e_vec = (e_vec 
                + np.matmul(e_vec.reshape(1,n), realOD_e)
                - e_vec * realOD_e.sum(axis=1))
            a_vec = a_vec + new_a - new_r_from_a
            a_vec = (a_vec 
                + np.matmul(a_vec.reshape(1,n), realOD_a)
                - a_vec * realOD_a.sum(axis=1))
            i_vec = i_vec + new_i - new_q
            i_vec = (i_vec 
                + np.matmul(i_vec.reshape(1,n), realOD_i)
                - i_vec * realOD_i.sum(axis=1))
            q_vec = q_vec + new_q - new_r_from_q - new_d
            q_vec = (q_vec 
                + np.matmul(q_vec.reshape(1,n), realOD_q)
                - q_vec * realOD_q.sum(axis=1))
            r_vec = r_vec + new_r_from_q + new_r_from_a + new_r_from_v - new_s
            r_vec = (r_vec 
                + np.matmul(r_vec.reshape(1,n), realOD_r)
                - r_vec * realOD_r.sum(axis=1))
            d_vec = d_vec + new_d
            v_vec = v_vec + new_v - new_r_from_v

            # Add the accumulated compartment values to results
            result[i + 1,:] = [s_vec.sum(), e_vec.sum(), a_vec.sum(), i_vec.sum(), q_vec.sum(), r_vec.sum(),  d_vec.sum(), h_vec.sum(), v_vec.sum()]
            
            # Add number of hospitalized 
            total_new_infected[i + 1] = new_i.sum()
            result[i + 1, 8] = total_new_infected[max(0, i - self.par.hospital_duration) : i].sum() * self.par.hospitalisation_rate
            
            # Add the accumulated compartment values to results
            history = self.update_history_results(s_vec, e_vec, a_vec, i_vec, q_vec, r_vec, d_vec, v_vec, h_vec, history, i)
        
        # write results to csv
        if write_to_csv:
            utils.write_history(write_weekly,
                                history, 
                                self.par.population, 
                                state.time_step, 
                                self.paths.results_weekly, 
                                self.paths.results_history,
                                compartments)
        
        state_compartments_values[age_group] = result, total_new_infected.sum(), history
        
        return state_compartments_values
    
    @staticmethod
    def add_hidden_cases(s_vec, i_vec, new_i):
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
    
    @staticmethod
    def get_relevant_flow(r, realflow_s, realflow_e, realflow_a, realflow_i, realflow_q, realflow_r, i):
        """ Gets relevant realflow value"""
        realOD_s = realflow_s[i % r]
        realOD_e = realflow_e[i % r]
        realOD_a = realflow_a[i % r]
        realOD_i = realflow_i[i % r]
        realOD_q = realflow_q[i % r]
        realOD_r = realflow_r[i % r]
        return realOD_s,realOD_e,realOD_a,realOD_i,realOD_q,realOD_r

    @staticmethod
    def update_history_results(s_vec, e_vec, a_vec, i_vec, q_vec, r_vec, d_vec, v_vec, h_vec, history, i):
        """ Updates history results with new compartments values"""
        history[i + 1,0,:] = s_vec
        history[i + 1,1,:] = e_vec
        history[i + 1,2,:] = a_vec
        history[i + 1,3,:] = i_vec
        history[i + 1,4,:] = q_vec
        history[i + 1,5,:] = r_vec
        history[i + 1,6,:] = d_vec
        history[i + 1,7,:] = v_vec
        history[i + 1,8,:] = h_vec
        return history

    @staticmethod
    def initialize_output_matrices(decision_period, k, n, s_vec, e_vec, a_vec, i_vec, q_vec, r_vec, d_vec, v_vec, h_vec):
        """ Initializes output matrices """
        history = np.zeros((decision_period, k, n))
        history[0,0,:] = s_vec
        history[0,1,:] = e_vec
        history[0,2,:] = a_vec
        history[0,3,:] = i_vec
        history[0,4,:] = q_vec
        history[0,5,:] = r_vec
        history[0,6,:] = d_vec
        history[0,7,:] = v_vec
        history[0,8,:] = h_vec

        result = np.zeros((decision_period, k))
        result[0,:] = [s_vec.sum(), e_vec.sum(), a_vec.sum(), i_vec.sum(), q_vec.sum(), r_vec.sum(), d_vec.sum(), v_vec.sum(), 0]
        return history, result


    
    def get_realflows(self, alphas):
        """ Gets realflows after each flow is scaled with a respective alpha value"""
        alpha_s, alpha_e, alpha_a, alpha_i, alpha_q, alpha_r = alphas
        realflow_s = self.scale_flow(alpha_s)
        realflow_e = self.scale_flow(alpha_e)
        realflow_a = self.scale_flow(alpha_a)
        realflow_i = self.scale_flow(alpha_i)
        realflow_q = self.scale_flow(alpha_q)
        realflow_r = self.scale_flow(alpha_r)
        return realflow_s,realflow_e,realflow_a,realflow_i,realflow_q,realflow_r

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


    def generate_contact_matrix(self, weights):
        """ scale the contact matrices  with weights, and return the master contact matriz

        """
        # NEED TO UPDATE
        return self.par.contact_matrices[0]
