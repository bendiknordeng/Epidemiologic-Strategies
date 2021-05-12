import numpy as np
from utils import generate_weighted_contact_matrix

class Policy:
    def __init__(self, config, policy, population, contact_matrices, age_flow_scaling, GA):
        """ Defining vaccine allocation pollicy

        Args:
            config (namedtuple): case specific data
            policy (str): name of the vaccine allocation policy to be used
            population (pandas.DataFrame): information about population in reions and age groups
        """
        self.config = config
        self.policy_name = policy
        self.policies = {
            "random": self._random_policy,
            "no_vaccines": self._no_vaccines,
            "susceptible_based": self._susceptible_based_policy,
            "infection_based": self._infection_based_policy,
            "oldest_first": self._oldest_first_policy,
            "contact_based": self._contact_based_policy,
            "commuter_based": self._commuter_based_policy,
            "weighted": self._weighted_policy
            }
        self.vaccine_allocation = self.policies['weighted'] if GA else self.policies[policy]
        self.population = population
        self.contact_matrices = contact_matrices
        self.age_flow_scaling = age_flow_scaling
        self.GA = GA

    def get_decision(self, state, vaccines, weights):
        """ Retrieves a vaccine allocation

        Args:
            state (State): current state in the simulation
            vaccines (int): number of available vaccines
            weights (numpy.ndarray): weights for each policy if weighted policy

        Returns:
            numpy.ndarray: vaccine allocation given the state, vaccines available and policy_weights (#regions, #age_groups)
        """
        if state.vaccines_available == 0:
            return self._no_vaccines()
        return self.vaccine_allocation(state, vaccines, weights)

    def _random_policy(self, state, vaccines):
        """ Define allocation of vaccines based on a random distribution

        Returns
            numpy.ndarray: vaccine allocation of shape (#regions, #age_groups)
        """
        n_regions, n_age_groups = self.population.shape
        vaccine_allocation = np.zeros((n_regions, n_age_groups))
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        M = vaccines
        while M > 0:
            possible_regions = np.nonzero(demand > 0)[0]
            region = np.random.choice(possible_regions)
            possible_age_groups = np.nonzero(demand[region] > 0)[0]
            age_group = np.random.choice(possible_age_groups)
            allocation = np.min([M, demand[region][age_group], 1]) # consider fractional populations
            M -= allocation
            vaccine_allocation[region][age_group] += allocation
            demand[region][age_group] -= allocation
        decision = np.minimum(demand, vaccine_allocation).clip(min=0)
        return decision

    def _no_vaccines(self, *args):
        """ Define allocation of vaccines to zero

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        return np.zeros(self.population.shape)

    def _susceptible_based_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on number of susceptibles in each region

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        vaccine_allocation = np.zeros(self.population.shape)
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        M = vaccines
        if M > 0:
            vaccine_allocation = M * demand/np.sum(demand)
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _infection_based_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on number of infected in each region

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        vaccine_allocation = np.zeros(self.population.shape)
        total_infection = np.sum(state.I)
        M = vaccines
        if M > 0:
            if total_infection > 0:
                demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
                infection_density = state.I.sum(axis=1)/total_infection
                regional_allocation = M * infection_density
                total_regional_demand = demand.sum(axis=1).reshape(-1,1)
                vaccine_allocation = demand * regional_allocation.reshape(-1,1)/np.where(total_regional_demand==0, np.inf, total_regional_demand) 
                decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                return decision
        return vaccine_allocation

    def _oldest_first_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on age, prioritizes the oldest non-vaccinated group

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        vaccine_allocation = np.zeros(self.population.shape)
        M = vaccines
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            for age_group in range(self.population.shape[1]-1,0,-1):
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                    return decision
                else:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _contact_based_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on amount of contact, prioritize age group with most contact

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        C = generate_weighted_contact_matrix(self.contact_matrices, self.config.initial_contact_weights)
        contact_sum = C.sum(axis=1)
        priority = sorted(zip(range(len(contact_sum)), contact_sum), key=lambda x: x[1], reverse=True)
        vaccine_allocation = np.zeros(self.population.shape)
        M = vaccines
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            for age_group in tuple(zip(*priority))[0]:
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                    return decision
                else:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _commuter_based_policy(self, state, vaccines, *args):   
        """ Define allocation of vaccines based on amount of contact, prioritize age group with most contact

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        priority = sorted(zip(range(len(self.age_flow_scaling)), self.age_flow_scaling), key=lambda x: x[1], reverse=True)
        vaccine_allocation = np.zeros(self.population.shape)
        M = vaccines
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            for age_group in tuple(zip(*priority))[0]:
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                    return decision
                else:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _weighted_policy(self, state, vaccines, weights):
        """ Define allocation of vaccines based on a weighting of other policies

        Returns:
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        if weights is None:
            return self._no_vaccines()
        if self.GA:
            i = {"U": 0, "D": 1, "N": 2}[state.wave_state]
            j = min(state.wave_count[state.wave_state], 2) # make sure strategy is kept within count 3
            weights = weights[i][j-1]
            weighted_policies = ["no_vaccines", "susceptible_based", "infection_based", "oldest_first"]
        else: 
            weighted_policies = ["no_vaccines", "susceptible_based", "infection_based", "oldest_first", "contact_based", "commuter_based"]
        vaccine_allocation = np.zeros(self.population.shape)
        M = vaccines
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            vaccines_per_policy = M * weights
            for i, policy in enumerate(weighted_policies):
                vaccine_allocation += self.policies[policy](state, vaccines_per_policy[i])
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def __str__(self):
        return self.policy_name
    
    def __repr__(self):
        return self.policy_name