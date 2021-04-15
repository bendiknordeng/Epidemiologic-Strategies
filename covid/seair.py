import numpy as np
from covid import utils
np.random.seed(10)

class SEAIR:
    def __init__(self, OD, population, config, paths, include_flow, hidden_cases, write_to_csv, write_weekly):
        """ 
        Parameters:
            OD: Origin-Destination matrix
            population: pd.DataFrame with columns region_id, region_name, population (quantity)
            config: named tuple with following parameters
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
            include_flow: boolean, true if we want to model population flow between regions
            hidden_cases: boolean, true if we want to model hidden cases of infection
            write_to_csv: boolean, true if we want to write results to csv
            write_weekly: boolean, false if we want to write daily results, true if weekly
         """
        
        self.periods_per_day = int(24/config.time_delta)
        self.OD=OD
        self.population=population
        self.contact_matrices=config.contact_matrices
        self.age_group_flow_scaling=config.age_group_flow_scaling
        self.R0=config.R0*self.periods_per_day
        self.efficacy=config.efficacy
        self.latent_period=config.latent_period*self.periods_per_day
        self.proportion_symptomatic_infections=config.proportion_symptomatic_infections
        self.presymptomatic_infectiousness=config.presymptomatic_infectiousness
        self.asymptomatic_infectiousness=config.asymptomatic_infectiousness
        self.presymptomatic_period=config.presymptomatic_period*self.periods_per_day
        self.postsymptomatic_period=config.postsymptomatic_period*self.periods_per_day
        self.recovery_period = self.presymptomatic_period + self.postsymptomatic_period
        self.fatality_rate_symptomatic=config.fatality_rate_symptomatic

        self.include_flow = include_flow
        self.hidden_cases = hidden_cases
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

        # Scale movement flows with alphas (movement restrictions for each region and compartment)
        alphas = information['alphas']
        realflows = [self.OD.copy()*a for a in alphas]

        # Initialize history matrix and total new infected
        total_new_infected = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        history = np.zeros(shape=(int(decision_period/self.periods_per_day), n_compartments+1, n_regions, n_age_groups))
        
        # Define parameters in the mathematical model
        N = self.population.population.to_numpy(dtype='float64')
        beta = (self.R0/self.recovery_period)
        sigma = 1/self.latent_period
        p = self.proportion_symptomatic_infections
        r_e = self.presymptomatic_infectiousness
        r_a = self.asymptomatic_infectiousness
        alpha = 1/self.presymptomatic_period
        omega = 1/self.postsymptomatic_period
        gamma = 1/self.recovery_period
        delta = self.fatality_rate_symptomatic
        epsilon = self.efficacy
        C = self.generate_weighted_contact_matrix(information['contact_matrices_weights'])

        # Run simulation
        S, E1, E2, A, I, R, D, V = compartments
        for i in range(0, decision_period-1):
            # Vaccinate before flow
            new_V = epsilon * decision[i % decision_period] # M
            S = S - new_V
            R = R + new_V
            V = V + new_V

            # Finds the movement flow for the given period i and scales it for each 
            if self.include_flow:
                realODs = [r[i % decision_period] for r in realflows]
                total_population = np.sum(N)
                flow_compartments = [S, E1, E2, A, I]
                for ix, compartment in enumerate(flow_compartments):
                    comp_pop = np.sum(compartment)
                    realODs[ix] = realODs[ix] * comp_pop/total_population if comp_pop > 0 else realODs[ix]*0
                    if i % 2 == 1:
                        flow_compartments[ix] = self.flow_transition(compartment, realODs[ix])
                
                S, E1, E2, A, I = flow_compartments

            draws = S.astype(int)
            prob_e = (np.matmul(E2, C).T * (r_e*beta/N)).T
            prob_a = (np.matmul(A, C).T * (r_a*beta/N)).T
            prob_i = (np.matmul(I, C).T * (beta/N)).T

            new_E1_from_E2 = np.random.binomial(draws, prob_e)
            new_E1_from_A = np.random.binomial(draws, prob_a)
            new_E1_from_I = np.random.binomial(draws, prob_i)

            new_E1 = new_E1_from_E2 + new_E1_from_A + new_E1_from_I

            if self.hidden_cases and (i % (decision_period/7) == 0): # Add random infected to new E if hidden_cases=True
                new_E1 += self.add_hidden_cases(S, E1, new_E1)

            new_E2 = p * sigma * E1
            new_A = (1 - p) * sigma * E1
            new_I = alpha * E2
            new_R_from_A = gamma * A
            new_R_from_I = I * (np.ones(len(delta)) - delta) * omega
            new_D = I * delta * omega

            # Calculate values for each compartment
            S = S - new_E1
            E1 = E1 + new_E1 - new_E2 - new_A
            E2 = E2 + new_E2 - new_I
            A = A + new_A - new_R_from_A
            I = I + new_I - new_R_from_I
            R = R + new_R_from_I + new_R_from_A
            D = D + new_D
           
            # Save number of new infected
            total_new_infected[i] = new_E1
            
            # Append simulation results
            if i%self.periods_per_day == 0: # record daily history
                daily_new_infected = new_E1 if i == 0 else total_new_infected[i-self.periods_per_day:i].sum(axis=0)
                history = self.update_history([S, E1, E2, A, I, R, D, V], daily_new_infected, history, i//self.periods_per_day)

            # Ensure balance in population
            # comp_pop = np.sum(S+ E1 + E2 + A + I + R + D)
            # total_pop = np.sum(N)
            # assert round(comp_pop) == total_pop, f"Population not in balance. \nCompartment population: {comp_pop}\nTotal population: {total_pop}"

            # Ensure all positive compartments
            compartment_labels = ['S', 'E1', 'E2', 'A', 'I']
            for index, c in enumerate([S, E1, E2, A, I]):
                assert round(np.min(c)) >= 0, f"Negative value in compartment {compartment_labels[index]}: {np.min(c)}"

        if self.write_to_csv:
            utils.write_history(self.write_weekly,
                                history,
                                self.population, 
                                state.time_step,
                                self.paths.results_weekly, 
                                self.paths.results_history,
                                ['S', 'E1', 'E2', 'A', 'I', 'R', 'D', 'V', 'New infected'])
        
        return S, E1, E2, A, I, R, D, V, total_new_infected.sum(axis=0)
    
    @staticmethod
    def add_hidden_cases(S, E1, new_E1):
        """ Adds cases to the infection compartment, to represent hidden cases

        Parameters
            S: array of susceptible in each region for each age group
            I: array of infected in each region for each age group
            new_I: array of new cases of infected individuals for each region and age group
        Returns
            new_I, an array of new cases including hidden cases
        """
        share = 0.1 # maximum number of hidden infections
        for i in range(len(E1)):
            for j in range(len(E1[0])):
                if E1[i][j] < 0.5:
                    new_infections = np.random.uniform(0, 0.01) # introduce infection to region with little infections
                else:
                    new_infections = np.random.randint(0, min(int(E1[i][j]*share), 10)+1)
                if S[i][j] > new_infections:
                    new_E1[i][j] += new_infections

        return new_E1

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
        total = compartment.sum(axis=1)
        inflow = np.matmul(total, OD)
        outflow = total * OD.sum(axis=1)
        net_flow = [age_flow_scaling * x for x in (inflow-outflow)]
        new_compartment = compartment + net_flow

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
        C = self.contact_matrices
        return np.sum(np.array([np.array(C[i])*weights[i] for i in range(len(C))]), axis=0)
