from vaccine_allocation_model.State import State
import numpy as np
from tqdm import tqdm
np.random.seed(10)

class MarkovDecisionProcess:
    def __init__(self, OD_matrices, population, seaiqr, vaccine_supply, horizon, decision_period, policy, infection_boost):
        """ Initializes an instance of the class MarkovDecisionProcess, that administrates

        Parameters
            OD_matrices: Origin-Destination matrices giving movement patterns between regions
            population: A DataFrame with region_id, region_name and population
            seaiqr: A seaiqr model that enables simulation of the decision process
            vaccine_supply: Information about supply of vaccines, shape e.g. (#decision_period, #regions)
            horizon: The amount of decision_periods the decision process is run 
            decision_period: The number of time steps that every decision directly affects
            policy: How the available vaccines should be distributed.
        """
        self.horizon = horizon
        self.OD_matrices = OD_matrices
        self.population = population
        self.vaccine_supply = vaccine_supply
        self.seaiqr = seaiqr
        self.state = self._initialize_state(None, 1000, 1000, infection_boost=infection_boost)
        self.path = [self.state]
        self.decision_period = decision_period

        policies = {
            "no_vaccines": self._no_vaccines,
            "random": self._random_policy,
            "population_based": self._population_based_policy
        }

        self.policy = policies[policy]

    def run(self):
        """ Updates states from current time_step to a specified horizon

        Returns
            A path that shows resulting traversal of states
        """
        for _ in tqdm(range(self.state.time_step, self.horizon)):
            self.update_state()
            # print(np.sum(self.state.R), np.sum(self.population.population))
            if np.sum(self.state.R) / np.sum(self.population.population) > 0.9: # stop if recovered population is 70 % of total population
                break
        return self.path

    def get_exogenous_information(self, state):
        """ Recieves the exogenous information at time_step t

        Parameters
            t: time_step
            state: state that 
        Returns:
            returns a dictionary of information contain 'alphas', 'vaccine_supply', 'contact_matrices_weights'
        """
        infection_level = self.get_infection_level()
        alphas = self.get_alphas(infection_level)
        contact_matrices_weights = self.get_contact_matrices_weights(infection_level)
        information = {'alphas': alphas, 'vaccine_supply': self.vaccine_supply, 'contact_matrices_weights':contact_matrices_weights}
        return information

    def get_contact_matrices_weights(self, infection_level):
        """ Returns the weight for contact matrices based on compartment values 
        Returns 
            weights for contact matrices
        """
        # contact_matrices_weights = [0.31, 0.24, 0.16, 0.29]
        contact_matrices_weights = [1, 1, 1, 1]
        return contact_matrices_weights

    def get_alphas(self, infection_level):
        """ Scales alphas with a given weight for each compartment
        Returns 
            alphas scaled with a weight for each compartment
        """
        alphas = [1, 1, 1, 0.05] # movement for compartments S,E,A,I
        return alphas
    
    def get_infection_level(self):
        """ Decide what infection level every region is currently at
        Returns
            integer indicating current infection level each region and age group on a scale from 1-3, 3 being most severe
        """
        S, E, A, I, Q, R, D, V = self.state.get_compartments_values()
        pop_100k = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")/1e5
        I_per_100k = I/pop_100k
        # np.zeros_like(x)
        # print(f'Max:{np.max(I_per_100k)}')
        # print(f'Min:{np.min(I_per_100k)}')
        # import pdb; pdb.set_trace()
        # TO DO: logic to find infection level
        # calculate I_per_100K per region
        # I_per_100k = 1e5*I/population
        # 0-50 - level 1
        # 50-100 - level 2
        # >100 - level 3
        return 1

    def update_state(self, decision_period=28):
        """ Updates the state of the decision process.

        Parameters
            decision_period: number of periods forward in time that the decision directly affects
        """
        decision = self.policy()
        information = self.get_exogenous_information(self.state)
        self.state = self.state.get_transition(decision, information, self.seaiqr.simulate, decision_period)
        self.path.append(self.state)

    def _initialize_state(self, initial_infected, num_initial_infected, vaccines_available, infection_boost, time_step=0):
        """ Initializes a state, default from the moment a disease breaks out

        Parameters
            initial_infected: array of initial infected (1,356)
            num_initial_infected: number of infected persons to be distributed randomly across regions if initiaL_infected=None e.g 50
            vaccines_available: int, number of vaccines available at time
            infection_boost: array of initial infection boost for each age group
            time_step: timestep in which state is initialized. Should be in the range of (0, (24/time_timedelta)*7 - 1)
        Returns
            an initialized State object, type defined in State.py
        """
        # pop = self.population.population.to_numpy(dtype='float64')
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions = pop.shape[0]
        n_age_groups = pop.shape[1]
        S = pop.copy()
        E = np.zeros(pop.shape)
        A = np.zeros(pop.shape)
        I = np.zeros(pop.shape)
        Q = np.zeros(pop.shape)
        R = np.zeros(pop.shape)
        D = np.zeros(pop.shape)
        V = np.zeros(pop.shape)

        # Boost infected in Oslo
        I[0] += infection_boost
        S[0] -= infection_boost
        num_initial_infected -= sum(infection_boost)

        if initial_infected is None:
            initial = np.zeros(pop.shape)
            for i in range(num_initial_infected):
                loc = (np.random.randint(0, n_regions), np.random.randint(0, n_age_groups))
                if (S[loc[0]][loc[1]] > initial[loc[0]][loc[1]]):
                    initial[loc[0]][loc[1]] += 1.0
        else:
            initial = initial_infected
        assert ((S < initial).sum() == 0)

        S -= initial
        I += initial

        return State(S, E, A, I, Q, R, D, V, vaccines_available, time_step) 

    def _no_vaccines(self):
        """ Define allocation of vaccines to zero

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions, n_age_groups = pop.shape
        return np.zeros(shape=(self.decision_period, n_regions, n_age_groups))

    def _random_policy(self):
        """ Define allocation of vaccines based on random distribution

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        pop = self.population[self.population.columns[2:-1]].to_numpy(dtype="float64")
        n_regions, n_age_groups = pop.shape
        vaccine_allocation = np.array([np.zeros(pop.shape) for _ in range(self.decision_period)])
        demand = self.state.S.copy()
        vacc_available = self.state.vaccines_available
        while vacc_available > 0:
            period, region, age_group = np.random.randint(self.decision_period), np.random.randint(n_regions), np.random.randint(n_age_groups)
            if demand[region][age_group] > 100: 
                vacc_available -= 1
                vaccine_allocation[period][region][age_group] += 1
                demand[region][age_group] -= 1

        return vaccine_allocation

    def _population_based_policy(self):
        """ Define allocation of vaccines based on number of inhabitants in each region

        Returns
            a vaccine allocation of shape (#decision periods, #regions, #age_groups)
        """
        vaccine_allocation = []
        for period in range(self.decision_period):
            total_allocation = self.state.vaccines_available * self.state.S/np.sum(self.state.S)
            vaccine_allocation.append(total_allocation/self.decision_period)
        return vaccine_allocation


class Decision:
    def __init__(self):
        self.region_allocation = {}

    def allocate_to_region(self, region, age_allocation):
        self.region_allocation[region] = age_allocation

    @staticmethod
    def get_age_allocation(self, allocation):
        return {
            '0-5':   allocation[0],
            '6-15':  allocation[1],
            '16-19': allocation[2],
            '20-66': allocation[3],
            '67+':   allocation[4]
        }

    def __str__(self):
        return f""
