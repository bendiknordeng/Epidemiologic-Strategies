import pandas as pd
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
            "weighted": self._weighted_policy,
            "fhi_policy": self._fhi_policy,
            }
        self.vaccine_allocation = self.policies[policy]
        self.population = population
        self.contact_matrices = contact_matrices
        self.age_flow_scaling = age_flow_scaling
        self.fhi_vaccine_plan = None
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

    def _random_policy(self, state, M, *args):
        """ Define allocation of vaccines based on a random distribution

        Returns
            numpy.ndarray: vaccine allocation of shape (#regions, #age_groups)
        """
        n_regions, n_age_groups = self.population.shape
        vaccine_allocation = np.zeros((n_regions, n_age_groups))
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        while M > 0:
            possible_age_groups = np.delete(np.nonzero(demand.sum(axis=0) > 0)[0], 0)
            age_group = np.random.choice(possible_age_groups)
            possible_regions = np.nonzero(demand[:,age_group] > 0)[0]
            region = np.random.choice(possible_regions)
            allocation = np.min([M, demand[region][age_group], 100]) # consider fractional populations
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

    def _susceptible_based_policy(self, state, M, *args):
        """ Define allocation of vaccines based on number of susceptibles in each region

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        demand = demand[:,1:] # remove first age group
        vaccine_allocation = np.zeros(demand.shape)
        if M > 0:
            vaccine_allocation = M * demand/np.sum(demand)
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            decision = np.insert(decision, 0, 0, axis=1)
            return decision
        vaccine_allocation = np.insert(vaccine_allocation, 0, 0, axis=1)
        return vaccine_allocation

    def _infection_based_policy(self, state, M, *args):
        """ Define allocation of vaccines based on number of infected in each region

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        vaccine_allocation = np.zeros(self.population.shape)
        total_infection = np.sum(state.I)
        if M > 0:
            if total_infection > 0:
                demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
                demand = demand[:,1:] # remove first age group
                infection_density = state.I.sum(axis=1)/total_infection
                regional_allocation = M * infection_density
                total_regional_demand = demand.sum(axis=1).reshape(-1,1)
                vaccine_allocation = demand * regional_allocation.reshape(-1,1)/np.where(total_regional_demand==0, np.inf, total_regional_demand) 
                decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                decision = np.insert(decision, 0, 0, axis=1)
                return decision
        return vaccine_allocation

    def _oldest_first_policy(self, state, M, *args):
        """ Define allocation of vaccines based on age, prioritizes the oldest non-vaccinated group

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        demand = demand[:,1:] # remove first age group
        vaccine_allocation = np.zeros(demand.shape)
        if M > 0:
            for age_group in range(demand.shape[1]-1,0,-1):
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    break
                else:
                    vaccine_allocation[:,age_group] = age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            decision = np.insert(decision, 0, 0, axis=1)
            return decision
        vaccine_allocation = np.insert(vaccine_allocation, 0, 0, axis=1)
        return vaccine_allocation

    def _contact_based_policy(self, state, M, *args):
        """ Define allocation of vaccines based on amount of contact, prioritize age group with most contact

        Returns
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
        vaccine_allocation = np.zeros(demand.shape)
        C = generate_weighted_contact_matrix(self.contact_matrices, state.contact_weights)[1:]
        contact_sum = C.sum(axis=1)
        priority = sorted(zip(range(1,len(contact_sum)), contact_sum), key=lambda x: x[1], reverse=True)
        if M > 0:
            for age_group in tuple(zip(*priority))[0]:
                age_group_demand = demand[:,age_group]
                total_age_group_demand = np.sum(age_group_demand)
                if M < total_age_group_demand:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    break
                else:
                    vaccine_allocation[:,age_group] = M * age_group_demand/total_age_group_demand
                    M -= total_age_group_demand
                    demand[:,age_group] -= age_group_demand
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _weighted_policy(self, state, M, weights):
        """ Define allocation of vaccines based on a weighting of other policies

        Returns:
            numpy.ndarray: a vaccine allocation of shape (#regions, #age_groups)
        """
        if weights is None:
            return self._no_vaccines()
        if self.GA:
            trend = {"U": 0, "D": 1, "N": 2}[state.trend]
            trend_count = min(state.trend_count[state.trend], 3) # make sure strategy is kept within count 3
            weights = weights[trend][trend_count-1]
        weighted_policies = ["no_vaccines", "susceptible_based", "infection_based", "oldest_first", "contact_based"]
        vaccine_allocation = np.zeros(self.population.shape)
        if M > 0:
            demand = state.S.copy() - (1-self.config.efficacy) * state.V.copy()
            vaccines_per_policy = M * weights
            for i, policy in enumerate(weighted_policies):
                vaccine_allocation += self.policies[policy](state, vaccines_per_policy[i])
            decision = np.minimum(demand, vaccine_allocation).clip(min=0)
            return decision
        return vaccine_allocation

    def _fhi_policy(self, state, M, *args):
        vaccine_allocation = np.zeros(self.population.shape)
        if M > 0:
            demand = state.S.copy()-(1-self.config.efficacy)*state.V.copy()
            if np.sum(self.fhi_vaccine_plan['n_people']) > 0:
                while M > 0:
                    for p in range(len(self.fhi_vaccine_plan)):
                        priority = self.fhi_vaccine_plan.iloc[p]
                        vaccinations = priority[1]
                        age_group_ixs = np.where(priority[2:].values > 0)[0]
                        vaccinations_per_age_group = vaccinations/len(age_group_ixs)
                        vaccines_per_age_group = M/len(age_group_ixs)
                        allocation = np.minimum(vaccinations_per_age_group, vaccines_per_age_group)
                        age_group_pop = demand[:,age_group_ixs].sum(axis=0)
                        vaccine_allocation[:,age_group_ixs] += allocation * demand[:,age_group_ixs]/age_group_pop
                        priority[1] -= np.sum(allocation)
                        M -= np.sum(allocation)
                        if M == 0: break
                    self.fhi_vaccine_plan = self.fhi_vaccine_plan[self.fhi_vaccine_plan.n_people > 0]
                decision = np.minimum(demand, vaccine_allocation).clip(min=0)
                return decision
        return vaccine_allocation

    def __str__(self):
        return self.policy_name
    
    def __repr__(self):
        return self.policy_name