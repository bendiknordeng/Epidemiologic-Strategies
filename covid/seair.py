import numpy as np
from covid import utils

class SEAIR:
    def __init__(self, OD, contact_matrices, population, age_group_flow_scaling, death_rates,
                config, paths, include_flow, include_waves, stochastic, write_to_csv, write_weekly):
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
        self.OD = OD
        self.contact_matrices = contact_matrices
        self.population = population
        self.age_group_flow_scaling = age_group_flow_scaling
        self.fatality_rate_symptomatic = death_rates
        self.R0 = config.R0 * self.periods_per_day
        self.efficacy = config.efficacy
        self.latent_period = config.latent_period * self.periods_per_day
        self.proportion_symptomatic_infections = config.proportion_symptomatic_infections
        self.presymptomatic_infectiousness = config.presymptomatic_infectiousness
        self.asymptomatic_infectiousness = config.asymptomatic_infectiousness
        self.presymptomatic_period = config.presymptomatic_period*self.periods_per_day
        self.postsymptomatic_period = config.postsymptomatic_period*self.periods_per_day
        self.recovery_period = self.presymptomatic_period + self.postsymptomatic_period

        self.stochastic = stochastic
        self.include_flow = include_flow
        self.include_waves = include_waves
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
        compartments = state.get_compartments_values()
        n_compartments = len(compartments)
        n_regions, n_age_groups = compartments[0].shape

        # Get information data
        wave_factor = information['wave'] if self.include_waves else 1
        alphas = information['alphas']
        C = self.generate_weighted_contact_matrix(information['contact_weights'])
    
        # Initialize variables for saving history
        total_new_infected = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        history = np.zeros(shape=(int(decision_period/self.periods_per_day), n_compartments+1, n_regions, n_age_groups))
        
        # Define parameters in the mathematical model
        N = self.population.population.to_numpy(dtype='float64')
        beta = (self.R0/self.recovery_period) * wave_factor
        sigma = 1/self.latent_period
        p = self.proportion_symptomatic_infections
        r_e = self.presymptomatic_infectiousness
        r_a = self.asymptomatic_infectiousness
        alpha = 1/self.presymptomatic_period
        omega = 1/self.postsymptomatic_period
        gamma = 1/self.recovery_period
        delta = self.fatality_rate_symptomatic
        epsilon = self.efficacy
        

        # Run simulation
        S, E1, E2, A, I, R, D, V = compartments
        for i in range(0, decision_period-1):
            timestep = (state.date.weekday() + i//4) % decision_period
            
            # Vaccinate before flow
            new_V = decision/decision_period
            successfully_new_V = epsilon * new_V
            S = S - successfully_new_V
            R = R + successfully_new_V
            V = V + new_V

            # Finds the movement flow for the given period i and scales it for each 
            if self.include_flow:
                total_population = np.sum(N)
                flow_compartments = [S, E1, E2, A, I]
                for ix, compartment in enumerate(flow_compartments):
                    comp_pop = np.sum(compartment)
                    realOD = self.OD[timestep] * alphas[ix] * comp_pop/total_population
                    if timestep%2 == 1 and timestep < self.periods_per_day*5:
                        flow_compartments[ix] = self.flow_transition(compartment, realOD)
                
                S, E1, E2, A, I = flow_compartments

            if self.stochastic:
                # Get stochastic transitions
                prob_e = beta*r_e * np.matmul(C, (E2.T/N)).T
                prob_a = beta*r_a * np.matmul(C, (A.T/N)).T
                prob_i = beta * np.matmul(C, (I.T/N)).T
                new_E1 = np.random.binomial(S.astype(int), prob_e + prob_a + prob_i)
                new_E2 = np.random.binomial(E1.astype(int), sigma * p)
                new_A = np.random.binomial((E1-new_E2).astype(int), sigma * (1-p))
                new_I = np.random.binomial(E2.astype(int), alpha)
                new_R_from_A = np.random.binomial(A.astype(int), gamma)
                new_R_from_I = np.random.binomial(I.astype(int), (np.ones(len(delta)) - delta) * omega)
                new_D = np.random.binomial((I-new_R_from_I).astype(int), delta * omega)
            else:
                transition_e = beta*r_e * np.matmul(C, (E2.T/N)).T
                transition_a = beta*r_a * np.matmul(C, (A.T/N)).T
                transition_i = beta * np.matmul(C, (I.T/N)).T
                new_E1 = S * (transition_e + transition_a + transition_i)
                new_E2 = E1 * sigma * p
                new_A = E1 * sigma * (1-p)
                new_I = E2 * alpha
                new_R_from_A = A * gamma
                new_R_from_I = I * (np.ones(len(delta)) - delta) * omega
                new_D = I * delta * omega

            # Calculate values for each compartment
            S = S - new_E1
            E1 = E1 + new_E1 - new_E2 - new_A
            E2 = E2 + new_E2 - new_I
            A = A + new_A - new_R_from_A
            I = I + new_I - new_R_from_I - new_D
            R = R + new_R_from_I + new_R_from_A
            D = D + new_D

            # Save number of new infected
            total_new_infected[i] = new_I
            
            # Append simulation results
            if i%self.periods_per_day == 0: # record daily history
                daily_new_infected = new_E1 if i == 0 else total_new_infected[i-self.periods_per_day:i].sum(axis=0)
                history = self.update_history([S, E1, E2, A, I, R, D, V], daily_new_infected, history, i//self.periods_per_day)

            # Ensure balance in population
            comp_pop = np.sum(S + E1 + E2 + A + I + R + D)
            total_pop = np.sum(N)
            assert round(comp_pop) == total_pop, f"Population not in balance. \nCompartment population: {comp_pop}\nTotal population: {total_pop}"
            
        if self.write_to_csv:
            utils.write_history(self.write_weekly,
                                history,
                                self.population, 
                                state.time_step,
                                self.paths.results_history_weekly,
                                self.paths.results_history_daily,
                                ['S', 'E1', 'E2', 'A', 'I', 'R', 'D', 'V', 'New infected'])

        return S, E1, E2, A, I, R, D, V, total_new_infected.sum(axis=0)

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

    def flow_transition(self, compartment, OD):
        """ Calculates the netflow for every region and age group

        Parameters
            compartment: array of size (#regions, #age_groups) giving population for relevant compartment
            OD: scaled ODs for relevant compartment that use movement flow
        Returns
            an array of shape (#regions, #age groups) of net flows within each region and age group
        """
        age_flow_scaling = np.array(self.age_group_flow_scaling)
        inflow = np.array([np.matmul(compartment[:,i], OD * age_flow_scaling[i]) for i in range(len(age_flow_scaling))]).T
        outflow = np.array([compartment[:,i] * OD.sum(axis=1) * age_flow_scaling[i] for i in range(len(age_flow_scaling))]).T
        return compartment + inflow - outflow
        
    def generate_weighted_contact_matrix(self, contact_weights):
        """ Scales the contact matrices with weights, and return the weighted contact matrix used in modelling

        Parameters
            weights: list of floats indicating the weight of each contact matrix for school, workplace, etc. 
        Returns
            weighted contact matrix used in modelling
        """
        C = self.contact_matrices
        return np.sum(np.array([np.array(C[i])*contact_weights[i] for i in range(len(C))]), axis=0)
