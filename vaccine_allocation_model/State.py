import numpy as np

class State:
    def __init__(demand, infected, vaccines, time):
        self.demand = demand
        self.infected = infected
        self.vaccines = vaccines
        self.time = time
        self.post_decision_state = False

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
