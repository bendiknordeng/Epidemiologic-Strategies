import numpy as np
from utils import generate_weighted_contact_matrix

class SEAIR:
    def __init__(self, commuters, contact_matrices, population, age_group_flow_scaling, 
                death_rates, config, include_flow, stochastic, use_waves):
        """ Compartmental simulation model 

        Args:
            commuters (numpy.ndarray): matrix giving number of individuals commuting between two regions (#regions, #age_groups)
            contact_matrices (numpy.ndarray): symmetric matrix giving average contact between age groups (#age_groups, #age_groups)
            population (pandas.DataFrame): information about population in reions and age groups
            age_group_flow_scaling (numpy.ndarray): scaling factors for commuting in each age group
            death_rates (nuumpy.ndarray): death probabilities for each age group
            config (namedtuple): case specific data
            include_flow (boolean): True if simulation should include flow
            stochastic (boolean): True if commuting and contact infection should be stochastic
            use_waves (boolean): True if wave logic should be modeled
        """
        self.R0 = config.R0
        self.periods_per_day = config.periods_per_day
        self.time_delta = config.time_delta
        self.commuters = commuters
        self.contact_matrices = contact_matrices
        self.population = population
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
        self.use_waves = use_waves

    def simulate(self, state, decision, decision_period, information):
        """Simulates the development of an epidemic as modelled by current parameters

        Args:
            state (State): State object with values for each compartment
            decision (numpy.ndarray): Vaccine allocation for each region and age group (#regions, #age_groups)
            decision_period (int): number of timesteps in the simulation
            information (dict): exogenous information
        Returns:
            np.ndarrays: compartmental values and new infected/dead
        """
        # Meta-parameters
        S, E1, E2, A, I, R, D, V = state.get_compartments_values()
        n_regions, n_age_groups = S.shape
        age_flow_scaling = np.array(self.age_group_flow_scaling)

        # Get information data
        if self.use_waves:
            R_eff = information['R'] * self.periods_per_day
        else:
            R_eff = self.R0 * self.periods_per_day
        alphas = information['alphas']
        C = generate_weighted_contact_matrix(self.contact_matrices, information['contact_weights'])
        visitors = self.commuters[0]
        commuters = self.commuters[1] * information['flow_scale']

        # Initialize variables for saving history
        total_new_infected = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        total_new_deaths = np.zeros(shape=(decision_period, n_regions, n_age_groups))
        
        # Probabilities
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
            new_V = np.nan_to_num(np.minimum(S, decision/decision_period)) # in case new infected during decision period
            unused_V = np.sum(new_V - decision/decision_period)
            state.vaccines_available += unused_V

            successfully_new_V = epsilon * new_V
            S = S - successfully_new_V
            R = R + successfully_new_V
            V = V + new_V

            # Update population to account for new deaths
            N = sum([S, E1, E2, A, I, R])

            # Calculate beta
            N_infectious = np.sum([E2, A, I])
            beta = R_eff * N_infectious/(np.sum(E2) * (1/omega + r_e/alpha) + np.sum(A) * r_a/gamma + np.sum(I) * 1/omega)
            beta_C = beta * C
            beta_C_W = beta * self.contact_matrices[2]
            # Calculate new infected from commuting
            commuter_cases = 0
            working_hours = timestep < (self.periods_per_day * 5) and timestep % self.periods_per_day == 2
            if self.include_flow and working_hours:
                # Define current transmission of infection with commuters
                infectious_commuters = np.matmul(commuters.T, (E2 + A + I)/N)
                infectious_commuters = np.array([infectious_commuters[:,a] * age_flow_scaling[a] for a in range(len(age_flow_scaling))]).T
                lam_j = np.clip(infectious_commuters/visitors, 0, 1)
                lam_j = np.matmul(lam_j, beta_C_W) # only use work matrix
                lam_j = np.array([lam_j[:,a] * age_flow_scaling[a] for a in range(len(age_flow_scaling))]).T
                commuter_cases = S/N * np.matmul(commuters, lam_j)
                if self.stochastic:
                    commuter_cases = np.random.poisson(commuter_cases)

            # Define current transmission of infection without commuters
            lam_i = np.clip(np.matmul(E2 + A + I, beta_C), 0, 1)
            contact_cases = S/N * lam_i
            if self.stochastic:
                contact_cases = np.random.poisson(contact_cases)
            
            # Get transition values
            new_E1  = np.clip(contact_cases + commuter_cases, None, S)
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

            # Save number of new infected and dead
            total_new_infected[i] = new_I
            total_new_deaths[i] = new_D

        return S, E1, E2, A, I, R, D, V, total_new_infected.sum(axis=0), total_new_deaths.sum(axis=0)
