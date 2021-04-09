import time
import numpy as np
from collections import namedtuple
from covid import utils
from random import uniform, randint

class SEAIQR:
    def __init__(self, OD, population, contact_matrices, age_group_flow_scaling, contact_matrices_weights, R0=2.4,
                efficacy=0.95,  proportion_symptomatic_infections=0.8, latent_period=5.1*4, recovery_period=21*4,
                pre_isolation_infection_period=4.6*4, post_isolation_recovery_period=16.4*4, fatality_rate_symptomatic=0.01*4):
        """ 
        Parameters
        - self.par: parameters {
                    OD: Origin-Destination matrix
                    population: pd.DataFrame with columns region_id, region_name, population (quantity)
                    contact_matrices: list of lists with measurement of the intensitity of social contacts among seven age-groups at households, schools, workplaces, and public/community
                    age_group_flow_scaling: list of scaling factors for flow of each age group
                    R0: Basic reproduction number (e.g 2.4)
                    efficacy: vaccine efficacy (e.g 0.95)
                    proportion_symptomatic_infections: Proportion of symptomatic infections(e.g 0.8)
                    latent_period: Time before vaccine is effective (e.g 5.1*4)
                    recovery_period: Time to recover from receiving the virus to not being  (e.g 21'4)
                    pre_isolation_infection_period: Pre-isolation infection period (e.g 4.6*4)
                    post_isolation_recovery_period: Post-isolation recovery period (e.g 16.4*4)
                    fatality_rate_symptomatic: Fatality rate for people that experience symptoms (e.g 0.01)
         """
        self.paths = utils.create_named_tuple('filepaths.txt')
        param = namedtuple('param', 'OD population contact_matrices age_group_flow_scaling contact_matrices_weights R0 efficacy proportion_symptomatic_infections latent_period recovery_period pre_isolation_infection_period post_isolation_recovery_period fatality_rate_symptomatic')
        self.par = param(
                        OD=OD,
                        population=population,
                        contact_matrices=contact_matrices,
                        age_group_flow_scaling=age_group_flow_scaling,
                        contact_matrices_weights=contact_matrices_weights,
                        R0=R0,
                        efficacy=efficacy,  
                        proportion_symptomatic_infections=proportion_symptomatic_infections, 
                        latent_period=latent_period, 
                        recovery_period=recovery_period,
                        pre_isolation_infection_period=pre_isolation_infection_period, 
                        post_isolation_recovery_period=post_isolation_recovery_period, 
                        fatality_rate_symptomatic=fatality_rate_symptomatic
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
        compartments = 'SEAIQRDV'
        n_compartments = len(compartments)
        S, E, A, I, Q, R, D, V = state.get_compartments_values()
        n_regions, n_age_groups = S.shape

        # Extraxt movement information and scale flow
        realflow_S, realflow_E, realflow_A, realflow_I, realflow_Q, realflow_R = self.get_realflows(information['alphas'])
        
        # Initialize output matrices
        history = self.initialize_output_matrices(decision_period, n_compartments, n_regions, n_age_groups, S, E, A, I, Q, R, D, V)
        total_new_infected = np.zeros(decision_period+1)
        # Run simulation
        for i in range(0, decision_period-1):
            # Finds the flow between regions for each compartment 
            realOD_S, realOD_E, realOD_A, realOD_I, realOD_Q, realOD_R = self.get_relevant_flow(decision_period, realflow_S, realflow_E, realflow_A, realflow_I, realflow_Q, realflow_R, i)
            age_group_flow_scaling = np.array(self.par.age_group_flow_scaling)
            
            # Flow between regions
            flow_S, flow_E, flow_A, flow_I, flow_Q, flow_R = self.flow_transition(realOD_S, realOD_E, realOD_A, realOD_I, realOD_Q, realOD_R, age_group_flow_scaling)
            
            S += flow_S
            E += flow_E
            A += flow_A
            I += flow_I
            Q += flow_Q
            R += flow_R

            # Finds the decision - number of vaccines to be allocated to each region for a specific time period
            M = decision[i % decision_period]
            N = self.par.population.population.to_numpy(dtype='float64')
            beta = (self.par.R0/self.par.recovery_period)
            sigma = 1/self.par.latent_period
            p = self.par.proportion_symptomatic_infections
            alpha = 1/self.par.pre_isolation_infection_period
            gamma = 1/self.par.recovery_period
            delta = self.par.fatality_rate_symptomatic
            omega = 1/self.par.post_isolation_recovery_period
            epsilon = self.par.efficacy

            # Calculate values for each arrow in epidemic model
            C = self.generate_contact_matrix(self.par.contact_matrices_weights)
            new_E = np.transpose(np.transpose(np.matmul(S, C) * (A + I)) * (beta / N))  # Need to change this to force of infection 
            new_A = (1 - p) * sigma * E
            new_I = p * sigma * E
        
            # Add random infected to new I if it is included in modelling (NEED TO BE ADJUSTED FOR AGE GROUP DIMENSIONS)
            if hidden_cases and (i % (decision_period/7) == 0): 
                new_I = self.add_hidden_cases(S, I, new_I)
            
            new_Q = alpha * I
            new_R_from_A = gamma * A
            new_R_from_Q = Q * (np.ones(len(delta)) - delta) * omega
            new_D = Q * delta * omega
            new_V = epsilon * M

            # Calculate values for each compartment
            S = S - new_V - new_E
            E = E + new_E - new_I - new_A
            A = A + new_A - new_R_from_A
            I = I + new_I - new_Q
            Q = Q + new_Q - new_R_from_Q - new_D
            R = R + new_R_from_Q + new_R_from_A + new_V
            D = D + new_D
            V = V + new_V
            
            # Save number of new infected
            total_new_infected[i + 1] = new_I.sum()
            
            # Add the accumulated compartment values to results
            history = self.update_history_results(S, E, A, I, Q, R, D, V, history, i)

            comp_pop = np.sum(S)+ np.sum(E) + np.sum(A) + np.sum(I) + np.sum(Q) + np.sum(R) + np.sum(D)
            total_pop = np.sum(N)
            assert round(comp_pop) == total_pop, f"Population not in balance. \nCompartment population: {comp_pop}\nTotal population: {total_pop}"

        # write results to csv
        if write_to_csv:
            utils.write_history(write_weekly,
                                history, 
                                self.par.population, 
                                state.time_step, 
                                self.paths.results_weekly, 
                                self.paths.results_history,
                                compartments)
        
        return S, E, A, I, Q, R, D, V, sum(total_new_infected)
    
    @staticmethod
    def add_hidden_cases(S, I, new_i):
        """ Adds cases to the infection compartment, to represent hidden cases

        Parameters
            S: array of susceptible in each region
            I: array of infected in each region
            new_i: array of new cases of infected individuals
        Returns
            new_i, an array of new cases including hidden cases
        """
        share = 0.1 # maximum number of hidden infections
        I = I.reshape(-1) # ensure correct shape
        S = S.reshape(-1) # ensure correct shape
        new_i = new_i.reshape(-1) # ensure correct shape
        for i in range(len(I)):
            if I[i] < 0.5:
                new_infections = uniform(0, 0.01) # introduce infection to region with little infections
            else:
                new_infections = randint(0, min(int(I[i]*share), 1))
            if S[i] > new_infections:
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
        return realOD_s, realOD_e, realOD_a, realOD_i, realOD_q, realOD_r

    @staticmethod
    def update_history_results(S, E, A, I, Q, R, D, V, history, i):
        """ Updates history results with new compartments values"""
        history[i + 1,0,:] = S
        history[i + 1,1,:] = E
        history[i + 1,2,:] = A
        history[i + 1,3,:] = I
        history[i + 1,4,:] = Q
        history[i + 1,5,:] = R
        history[i + 1,6,:] = D
        history[i + 1,7,:] = V
        return history

    @staticmethod
    def initialize_output_matrices(decision_period, k, n_regions, n_age_groups, S, E, A, I, Q, R, D, V):
        """ Initializes output matrices """
        history = np.zeros((decision_period, k, n_regions, n_age_groups))
        history[0,0,:] = S
        history[0,1,:] = E
        history[0,2,:] = A
        history[0,3,:] = I
        history[0,4,:] = Q
        history[0,5,:] = R
        history[0,6,:] = D
        history[0,7,:] = V
        return history

    def flow_transition(self, realOD_S, realOD_E, realOD_A, realOD_I, realOD_Q, realOD_R, age_group_flow_scaling):
        inflow = realOD_S.sum(axis=0)
        outflow = realOD_S.sum(axis=1)
        flow_S = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])

        inflow = realOD_E.sum(axis=0)
        outflow = realOD_E.sum(axis=1)
        flow_E = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])

        inflow = realOD_A.sum(axis=0)
        outflow = realOD_A.sum(axis=1)
        flow_A = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])

        inflow = realOD_I.sum(axis=0)
        outflow = realOD_I.sum(axis=1)
        flow_I = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])

        inflow = realOD_Q.sum(axis=0)
        outflow = realOD_Q.sum(axis=1)
        flow_Q = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])
        
        inflow = realOD_R.sum(axis=0)
        outflow = realOD_R.sum(axis=1)
        flow_R = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])

        return flow_S, flow_E, flow_A, flow_I, flow_Q, flow_R

    def get_realflows(self, alphas):
        """ Gets realflows after each flow is scaled with a respective alpha value"""
        alpha_s, alpha_e, alpha_a, alpha_i, alpha_q, alpha_r = alphas
        realflow_S = self.scale_flow(alpha_s)
        realflow_E = self.scale_flow(alpha_e)
        realflow_A = self.scale_flow(alpha_a)
        realflow_I = self.scale_flow(alpha_i)
        realflow_Q = self.scale_flow(alpha_q)
        realflow_R = self.scale_flow(alpha_r)
        return realflow_S, realflow_E, realflow_A, realflow_I, realflow_Q, realflow_R

    def scale_flow(self, alpha):
        """ Scales flow of individuals between regions

        Parameters
            alpha: array of scalers that adjust flows for a given compartment and region
        Returns
            realflow, scaled flow
        """
        realflow = self.par.OD.copy() 
        #realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis]  # Normalize flow
        realflow = alpha * realflow 
        return realflow


    def generate_contact_matrix(self, weights):
        """ scale the contact matrices  with weights, and return the master contact matriz

        """
        C = self.par.contact_matrices
        return np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
