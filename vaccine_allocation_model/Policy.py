import numpy as np

class Policy:
    def __init__(self, config, policy, population):
        self.config = config
        self.policy_name = policy
        self.vaccine_distribution = self._set_policy(policy)
        self.population = population

    def get_decision(self, state, vaccines):
        return self.vaccine_distribution(state, vaccines)

    def _set_policy(self, policy):
        return {
            "random": self._random_policy,
            "no_vaccines": self._no_vaccines,
            "susceptible_based": self._susceptible_based_policy,
            "infection_based": self._infection_based_policy,
            "oldest_first": self._oldest_first_policy,
            "weighted": self._weighted_policy
            }[policy]

    def _random_policy(self, state, vaccines):
        """ Define allocation of vaccines based on random distribution

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
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
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        return np.zeros(self.population.shape)

    def _susceptible_based_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on number of susceptible inhabitants in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
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
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        vaccine_allocation = np.zeros(self.population.shape)
        total_infection = np.sum(state.I)
        M = vaccines
        if M > 0:
            if total_infection > 0:
                demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
                infection_density = state.I.sum(axis=1)/total_infection
                regional_allocation = M * infection_density
                vaccine_allocation = demand * regional_allocation.reshape(-1,1)/demand.sum(axis=1).reshape(-1,1)
                decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                return decision
        return vaccine_allocation

    def _oldest_first_policy(self, state, vaccines, *args):
        """ Define allocation of vaccines based on age, prioritize the oldest group

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
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

    def _weighted_policy(self, state, vaccines, weights):
        vaccine_allocation = np.zeros(self.population.shape)
        weighted_policies = ["no_vaccines", "susceptible_based", "infection_based", "oldest_first"]
        M = vaccines
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            vaccines_per_policy = M * weights
            for i, policy in enumerate(weighted_policies):
                vaccine_allocation += self.policies[policy](M=vaccines_per_policy[i])
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def __str__(self):
        return self.policy_name
    
    def __repr__(self):
        return self.policy_name