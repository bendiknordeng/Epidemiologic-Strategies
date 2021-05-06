import numpy as np
from covid import utils

class SEAIR:
    def __init__(self, OD, contact_matrices, population, age_group_flow_scaling, death_rates,
                config, paths, include_flow, stochastic, write_to_csv, write_weekly):
        """ 
        Parameters:
            OD: Origin-Destination matrix
            contact_matrices: Contact matrices between age groups
            population: pd.DataFrame with columns region_id, region_name, population (quantity)
            config: named tuple with following parameters
                age_group_flow_scaling: list of scaling factors for flow of each age group
                R0: Basic reproduction number (e.g 2.4)
                efficacy: vaccine efficacy (e.g 0.95)
                proportion_symptomatic_infections: Proportion of symptomatic infections(e.g 0.8)
                latent_period: Time before vaccine is effective (e.g 5.1*4)
                recovery_period: Time to recover from receiving the virus to not being  (e.g 21'4)
                pre_isolation_infection_period: Pre-isolation infection period (e.g 4.6*4)
                post_isolation_recovery_period: Post-isolation recovery period (e.g 16.4*4)
                fatality_rate_symptomatic: Fatality rate for people that experience symptoms (e.g 0.01)
            include_flow: boolean, true if we want to model population flow between regions
            hidden_cases: boolean, true if we want to model hidden cases of infection
            write_to_csv: boolean, true if we want to write results to csv
            write_weekly: boolean, false if we want to write daily results, true if weekly
        """

        self.periods_per_day = config.periods_per_day
        self.time_delta = config.time_delta
        self.OD = OD
        self.contact_matrices = contact_matrices
        self.total_pop = population.population.sum()
        self.age_group_flow_scaling = age_group_flow_scaling
        self.fatality_rate_symptomatic = death_rates
        self.efficacy = config.efficacy
        self.latent_period = config.latent_period
        self.proportion_symptomatic_infections = config.proportion_symptomatic_infections
        self.presymptomatic_infectiousness = config.presymptomatic_infectiousness
        self.asymptomatic_infectiousness = config.asymptomatic_infectiousness
        self.presymptomatic_period = config.presymptomatic_period
        self.postsymptomatic_period = config.postsymptomatic_period
        self.recovery_period = self.presymptomatic_period + self.postsymptomatic_period
        self.stochastic = stochastic
        self.include_flow = include_flow
        self.paths = paths
        self.write_to_csv = write_to_csv
        self.write_weekly = write_weekly

    def simulate(self, state, decision, decision_period, information):
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
        S, E1, E2, A, I, R, D, V = state.get_compartments_values()
        n_regions, n_age_groups = S.shape

        # Get information data
        R = information['R']
        alphas = information['alphas']
        C = self.generate_weighted_contact_matrix(information['contact_weights'])
        flow_scale = information['flow_scale']
    
        # Initialize variables for saving history
        total_new_infected = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        total_new_deaths = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        
        # Probabilities
        beta = R/self.recovery_period
        r_e = self.presymptomatic_infectiousness
        r_a = self.asymptomatic_infectiousness
        p = self.proportion_symptomatic_infections
        delta = self.fatality_rate_symptomatic
        epsilon = self.efficacy
        
        # Rates
        sigma = 1/(self.latent_period * self.periods_per_day)
        alpha = 1/(self.presymptomatic_period * self.periods_per_day)
        omega = 1/(self.postsymptomatic_period * self.periods_per_day)
        gamma = 1/(self.recovery_period * self.periods_per_day)

        # Run simulation
        for i in range(decision_period):
            timestep = (state.date.weekday() * self.periods_per_day + i) % decision_period

            # Vaccinate before flow
            new_V = decision/decision_period
            successfully_new_V = epsilon * new_V
            S = S - successfully_new_V
            R = R + successfully_new_V
            V = V + new_V

            # Perform movement flow
            working_hours = timestep < (self.periods_per_day * 5) and ((i+3)%self.periods_per_day==0 or (i+1)%self.periods_per_day==0)
            if self.include_flow and working_hours:
                realOD = self.OD[timestep] * flow_scale
                S, E1, E2, A, I, R = self.flow_transition([S, E1, E2, A, I, R], realOD)
            
            # Update population to account for new deaths
            N = sum([S, E1, E2, A, I, R]).sum(axis=1)
            
            # Define current transmission of infection
            infection_e = np.matmul(beta * r_e * C * alphas[0], E2.T/N).T
            infection_a = np.matmul(beta * r_a * C * alphas[1], A.T/N).T
            infection_i = np.matmul(beta * C * alphas[2], I.T/N).T
            if self.stochastic:
                # Get stochastic transitions
                new_E1  = np.random.binomial(S.astype(int), infection_e + infection_a + infection_i)
                new_E2  = np.random.binomial(E1.astype(int), sigma * p)
                new_A   = np.random.binomial((E1-new_E2).astype(int), sigma * (1 - p))
                new_I   = np.random.binomial(E2.astype(int), alpha)
                new_R_A = np.random.binomial(A.astype(int), gamma)
                new_R_I = np.random.binomial(I.astype(int), (np.ones(len(delta)) - delta) * omega)
                new_D   = np.random.binomial((I-new_R_I).astype(int), delta * omega)
            else:
                # Get deterministic transitions
                new_E1  = S  * (infection_e + infection_a + infection_i)
                new_E2  = E1 * sigma * p
                new_A   = E1 * sigma * (1 - p)
                new_I   = E2 * alpha
                new_R_A = A  * gamma
                new_R_I = I  * (np.ones(len(delta)) - delta) * omega
                new_D   = I  * delta * omega

            # Calculate values for each compartment
            S  = S - new_E1
            E1 = E1 + new_E1 - new_E2 - new_A
            E2 = E2 + new_E2 - new_I
            A  = A + new_A - new_R_A
            I  = I + new_I - new_R_I - new_D
            R  = R + new_R_I + new_R_A
            D  = D + new_D

            # Save number of new infected
            total_new_infected[i] = new_I
            total_new_deaths[i] = new_D

        return S, E1, E2, A, I, R, D, V, total_new_infected.sum(axis=0), total_new_deaths.sum(axis=0)

    @staticmethod
    def update_history(compartments, new_infected, history, time_step):
        """ Updates history results with new compartments values
        Parameters
            compartments: list of each compartments for a given time step
            history: compartment values for each region, time step, and age group shape: (#decision_period,  #compartments, #regions, #age groups)
            time_step: int indicating current time step in simulation
        Returns
            updated history array
            
        """
        for c in range(len(compartments)):
            history[time_step, c,:] = compartments[c]
        history[time_step,-1,:] = new_infected
        return history

    def flow_transition(self, flow_comps, OD):
        """ Calculates the netflow for every region and age group

        Parameters
            compartment: array of size (#regions, #age_groups) giving population for relevant compartment
            OD: scaled ODs for relevant compartment that use movement flow
        Returns
            an array of shape (#regions, #age groups) of net flows within each region and age group
        """
        flow_pop = sum(flow_comps.copy())
        age_flow_scaling = np.array(self.age_group_flow_scaling)
        inflow = np.array([np.matmul(flow_pop[:,i], OD * age_flow_scaling[i]) for i in range(len(age_flow_scaling))]).T
        outflow = np.array([flow_pop[:,i] * OD.sum(axis=1) * age_flow_scaling[i] for i in range(len(age_flow_scaling))]).T
        flow_pop = flow_pop + inflow - outflow
        total_flow_pop = np.sum(flow_pop)
        comp_fractions = np.array([np.sum(comp)/total_flow_pop for comp in flow_comps])
        flow_comps = np.array([frac*flow_pop for frac in comp_fractions])
        return flow_comps
        
    def generate_weighted_contact_matrix(self, contact_weights):
        """ Scales the contact matrices with weights, and return the weighted contact matrix used in modelling

        Parameters
            weights: list of floats indicating the weight of each contact matrix for school, workplace, etc. 
        Returns
            weighted contact matrix used in modelling
        """
        C = self.contact_matrices
        return np.sum(np.array([np.array(C[i])*contact_weights[i] for i in range(len(C))]), axis=0)
