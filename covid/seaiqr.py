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
        compartments = state.get_compartments_values()
        n_compartments = len(compartments)
        n_regions, n_age_groups = compartments[0].shape

        # Scale movement flows with alphas (movement restrictions for each region and compartment)
        realflows = [self.par.OD.copy()*a for a in information['alphas']]  
        
        # Initialize history matrix
        history = np.zeros((decision_period, n_compartments, n_regions, n_age_groups))
        self.update_history(compartments, history, 0)

        # Initialize infected history matrix
        total_new_infected = np.zeros(decision_period+1)
        
        # Run simulation
        S, E, A, I, Q, R, D, V = compartments
        for i in range(0, decision_period-1):
            # Finds the movement flow for the given period i and scales it for each 
            realODs = [r[i % decision_period] for r in realflows]

            # Calculates netflows between regions for every age group
            flow_S, flow_E, flow_A, flow_I, flow_Q, flow_R = self.flow_transition(realODs, np.array(self.par.age_group_flow_scaling))
            
            # Update compartmet values with net movement flows for each region and age group
            S += flow_S
            E += flow_E
            A += flow_A
            I += flow_I
            Q += flow_Q
            R += flow_R

            # Define parameters in the mathematical model
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

            # Calculate values for each arrow in epidemic modelS.
            C = self.generate_contact_matrix(self.par.contact_matrices_weights)
            new_E = np.transpose(np.transpose(np.matmul(S, C) * (A + I)) * (beta / N)) 
            new_A = (1 - p) * sigma * E
            new_I = p * sigma * E
        
            # Add random infected to new I if it is included in modelling (NEED TO BE ADJUSTED FOR AGE GROUP DIMENSIONS)
            if hidden_cases and (i % (decision_period/7) == 0): 
                new_I = self.add_hidden_cases(S, I, new_I)
            
            # import pdb; pdb.set_trace() 
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
            
            # Append simulation results
            history = self.update_history([S, E, A, I, Q, R, D, V], history, i)

            # Ensure balance in population
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
    def add_hidden_cases(S, I, new_I):
        """ Adds cases to the infection compartment, to represent hidden cases

        Parameters
            S: array of susceptible in each region for each age group
            I: array of infected in each region for each age group
            new_I: array of new cases of infected individuals for each region and age group
        Returns
            new_I, an array of new cases including hidden cases
        """
        share = 0.1 # maximum number of hidden infections
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j] < 0.5:
                    new_infections = uniform(0, 0.01) # introduce infection to region with little infections
                else:
                    new_infections = randint(0, min(int(I[i][j]*share), 10))
                if S[i][j] > new_infections:
                    new_I[i][j] += new_infections
        return new_I

    @staticmethod
    def update_history(compartments, history, time_step):
        """ Updates history results with new compartments values"""
        for c in range(len(compartments)):
            history[time_step+1, c,:] = compartments[c]
        return history

    @staticmethod
    def flow_transition(realODs, age_group_flow_scaling):
        """ Calculates the netflow for a every age group

        Parameters
            realODs: scaled ODs for all compartments that use movement flow
        Returns
            a list of absolute flows for each region and agegroup for compartments with movement flows shape (#compartments, #regions, #age groups)
        """
        flows = []
        for od in realODs:
            inflow = od.sum(axis=0)
            outflow = od.sum(axis=1)
            flow = np.array([age_group_flow_scaling * x for x in (inflow-outflow)])
            flows.append(flow)
        return flows


    def generate_contact_matrix(self, weights):
        """ Scales the contact matrices with weights, and return the weighted contact matrix used in modelling

        Parameters
            weights: list of floats indicating the weight of each contact matrix for school, workplace, etc. 
        Returns
            weighted contact matrix used in modelling
        """
        C = self.par.contact_matrices
        return np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
