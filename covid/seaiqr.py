import numpy as np
from collections import namedtuple
from covid import utils
np.random.seed(10)

class SEAIQR:
    def __init__(self, OD, population, contact_matrices, age_group_flow_scaling, R0=2.4,
                efficacy=0.95,  proportion_symptomatic_infections=0.8, latent_period=5.1*4, recovery_period=21*4,
                pre_isolation_infection_period=4.6*4, post_isolation_recovery_period=16.4*4, fatality_rate_symptomatic=0.01*4, 
                model_flow=True, hidden_cases=True, write_to_csv=False, write_weekly=False):
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
        param = namedtuple('param', 'OD population contact_matrices age_group_flow_scaling R0 efficacy proportion_symptomatic_infections latent_period recovery_period pre_isolation_infection_period post_isolation_recovery_period fatality_rate_symptomatic')
        self.par = param(
                        OD=OD,
                        population=population,
                        contact_matrices=contact_matrices,
                        age_group_flow_scaling=age_group_flow_scaling,
                        R0=R0,
                        efficacy=efficacy,  
                        proportion_symptomatic_infections=proportion_symptomatic_infections, 
                        latent_period=latent_period, 
                        recovery_period=recovery_period,
                        pre_isolation_infection_period=pre_isolation_infection_period, 
                        post_isolation_recovery_period=post_isolation_recovery_period, 
                        fatality_rate_symptomatic=fatality_rate_symptomatic
                        )

        self.model_flow = model_flow
        self.hidden_cases = hidden_cases
        self.write_to_csv = write_to_csv
        self.write_weekly = write_weekly
    
    def simulate(self, state, decision, decision_period, information, write_to_csv=False, write_weekly=True):
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
            history: compartment values for each region, time step, and age group shape: (#decision_period,  #compartments, #regions, #age groups)
        """
        # Meta-parameters
        compartments = state.get_compartments_values()
        n_compartments = len(compartments)
        n_regions, n_age_groups = compartments[0].shape

        # Scale movement flows with alphas (movement restrictions for each region and compartment)
        alphas = information['alphas']
        realflows = [self.par.OD.copy()*a for a in alphas]

        # Initialize history matrix and total new infected
        history = np.zeros((decision_period, n_compartments, n_regions, n_age_groups))
        history = self.update_history(compartments, history, 0)
        total_new_infected = np.zeros(decision_period+1)
        
        # Define parameters in the mathematical model
        N = self.par.population.population.to_numpy(dtype='float64')
        beta = (self.par.R0/self.par.pre_isolation_infection_period)
        sigma = 1/self.par.latent_period
        p = self.par.proportion_symptomatic_infections
        alpha = 1/self.par.pre_isolation_infection_period
        gamma = 1/self.par.recovery_period
        delta = self.par.fatality_rate_symptomatic
        omega = 1/self.par.post_isolation_recovery_period
        epsilon = self.par.efficacy
        C = self.generate_weighted_contact_matrix(information['contact_matrices_weights'])

        # Run simulation
        S, E, A, I, Q, R, D, V = compartments
        for i in range(0, decision_period-1):
            # Vaccinate before flow
            new_V = epsilon * decision[i % decision_period] # M
            S -= new_V
            R += new_V

            # Finds the movement flow for the given period i and scales it for each 
            if self.model_flow:
                realODs = [r[i % decision_period] for r in realflows]
                total_population = np.sum(N)
                flow_compartments = [S, E, A, I]
                for ix, compartment in enumerate(flow_compartments):
                    comp_pop = np.sum(compartment)
                    realODs[ix] = realODs[ix] * comp_pop/total_population if comp_pop > 0 else realODs[ix]*0
                    if i % 2 == 1:
                        flow_compartments[ix] = self.flow_transition(compartment, realODs[ix])
                
                S, E, A, I = flow_compartments

            # Calculate values for each arrow in epidemic modelS.
            draws = S.astype(int)
            prob = (np.matmul((A+I), C).T * (beta/N)).T
            new_E = np.random.binomial(draws, prob)

            if self.hidden_cases and (i % (decision_period/7) == 0): # Add random infected to new E if hidden_cases=True
                new_E = self.add_hidden_cases(S, I, new_E)

            new_A = (1 - p) * sigma * E
            new_I = p * sigma * E
            new_Q = alpha * I
            new_R_from_A = gamma * A
            new_R_from_Q = Q * (np.ones(len(delta)) - delta) * omega
            new_D = Q * delta * omega

            # Calculate values for each compartment
            S = S - new_E
            E = E + new_E - new_I - new_A
            A = A + new_A - new_R_from_A
            I = I + new_I - new_Q
            Q = Q + new_Q - new_R_from_Q - new_D
            R = R + new_R_from_Q + new_R_from_A
            D = D + new_D
            V = V + new_V
            
            # Save number of new infected
            total_new_infected[i + 1] = np.sum([new_I, new_A])
            
            # Append simulation results
            history = self.update_history([S, E, A, I, Q, R, D, V], history, i)

            # Ensure balance in population
            comp_pop = np.sum(S)+ np.sum(E) + np.sum(A) + np.sum(I) + np.sum(Q) + np.sum(R) + np.sum(D)
            total_pop = np.sum(N)
            # assert round(comp_pop) == total_pop, f"Population not in balance. \nCompartment population: {comp_pop}\nTotal population: {total_pop}"

            # Ensure all positive compartments
            try:
                for c in [S, E, A, I, Q, R, D]:
                    assert round(np.min(c)) >= 0, f"Negative compartment values: {np.min(c)}"
            except AssertionError:
                import pdb; pdb.set_trace()

        # write results to csv
        if self.write_to_csv:
            utils.write_history(self.write_weekly,
                                history, 
                                self.par.population, 
                                state.time_step, 
                                self.paths.results_weekly, 
                                self.paths.results_history,
                                compartments)
        
        return S, E, A, I, Q, R, D, V, sum(total_new_infected)
    
    @staticmethod
    def add_hidden_cases(S, I, new_E):
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
                    new_infections = np.random.uniform(0, 0.01) # introduce infection to region with little infections
                else:
                    new_infections = np.random.randint(0, min(int(I[i][j]*share), 10)+1)
                if S[i][j] > new_infections:
                    new_E[i][j] += new_infections
        return new_E

    @staticmethod
    def update_history(compartments, history, time_step):
        """ Updates history results with new compartments values
        Parameters
            compartments: list of each compartments for a given time step
            history: compartment values for each region, time step, and age group shape: (#decision_period,  #compartments, #regions, #age groups)
            time_step: int indicating current time step in simulation
        Returns
            updated history array
            
        """
        for c in range(len(compartments)):
            history[time_step+1, c,:] = compartments[c]
        return history

    def flow_transition(self, compartment, OD):
        """ Calculates the netflow for every region and age group

        Parameters
            compartment: array of size (#regions, #age_groups) giving population for relevant compartment
            OD: scaled ODs for relevant compartment that use movement flow
        Returns
            an array of shape (#regions, #age groups) of net flows within each region and age group
        """
        age_flow_scaling = np.array(self.par.age_group_flow_scaling)
        total = compartment.sum(axis=1)
        inflow = np.array([age_flow_scaling * x for x in np.matmul(total, OD)])
        outflow = np.array([age_flow_scaling * x for x in total * OD.sum(axis=1)])
        new_compartment = compartment + inflow - outflow

        # fix negative values from rounding errors
        negatives = np.where(new_compartment < 0)
        for i in negatives[0]:
            for j in negatives[1]:
                new_compartment[i][j] = 0

        return new_compartment

    def generate_weighted_contact_matrix(self, weights):
        """ Scales the contact matrices with weights, and return the weighted contact matrix used in modelling

        Parameters
            weights: list of floats indicating the weight of each contact matrix for school, workplace, etc. 
        Returns
            weighted contact matrix used in modelling
        """
        C = self.par.contact_matrices
        return np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
