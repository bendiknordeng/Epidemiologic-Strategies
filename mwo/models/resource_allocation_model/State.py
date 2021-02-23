__author__ = "Martin Willoch Olstad"
__email__ = "martinwilloch@gmail.com"

import numpy as np
import torch


class State:

    def __init__(self, resources, demands, fatalities, time=0, dispersal=None, time_dependent_resources=None):
        self.resources = resources
        self.demands = demands
        self.fatalities = fatalities
        self.post_decision_state = False
        self.time = time

        self.time_dependent_resources = time_dependent_resources

        self.dispersal = dispersal

    def get_transition(self, decisions, information, epidemic_function, actual_transition=True, previous_decision=False, old=False):
        if previous_decision:
            resources = self.resources
        else:
            resources = self.resources - np.sum(decisions.astype(dtype='int64'), axis=0)
        if self.time_dependent_resources is not None and self.time_dependent_resources.get(str(self.time)) is not None:
            resources += self.time_dependent_resources[str(self.time)]

        new_epidemic_state = epidemic_function(decisions, information, dispersal=self.dispersal, time=self.time, actual_transition=actual_transition, old=old)

        susceptible_demand = new_epidemic_state[:, 0]
        infected_demand = new_epidemic_state[:, 2]

        demands = np.array([susceptible_demand, infected_demand], dtype='int64').transpose()
        fatalities = np.sum(new_epidemic_state[:, 5], axis=0)

        return State(resources=resources, demands=demands, fatalities=fatalities, time=self.time+1,
                     dispersal=self.dispersal, time_dependent_resources=self.time_dependent_resources)

    def transition(self, decisions, information, epidemic_function):
        self.decision_transition(decisions)
        self.information_transition(decisions, information, epidemic_function)
        self.time += 1

    def decision_transition(self, decisions):
        self.resources -= np.sum(decisions.astype(dtype='int64'), axis=0)
        self.post_decision_state = True

    def information_transition(self, decisions, information, epidemic_function):
        new_epidemic_state = epidemic_function(decisions, information, time=self.time)
        susceptible_demand = new_epidemic_state[:, 0]
        infected_demand = new_epidemic_state[:, 2]
        self.demands = np.array([susceptible_demand, susceptible_demand, infected_demand, infected_demand], dtype='int64').transpose()
        self.fatalities = new_epidemic_state[-1, 5]
        self.post_decision_state = False

    def to_flat_numpy(self):
        array = np.concatenate((self.resources, self.demands.flatten(order='C'),np.array([self.fatalities])))
        array.shape = (1,array.shape[0])
        return array

    def to_tensor(self):
        array = self.to_flat_numpy()
        tensor = torch.from_numpy(array).float()
        return tensor

    def to_string(self):
        print('Resources: ')
        print(self.resources)

        print('Demands: ')
        print(self.demands)

        print(f'Fatalities: {self.fatalities}')
        print()

