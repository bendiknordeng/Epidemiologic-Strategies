import numpy as np

class State:
    

    def __init__(self, S, E, I, R, H, V, vaccines_available, time_step):
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.H = H
        self.V = V
        self.vaccines_available = vaccines_available
        self.time_step = time_step
        self.new_infected = 0


    def get_transition(self, decision, information, epidemic_function, decision_period):
        result, new_infected, history = epidemic_function(self, decision, decision_period, information, write_to_csv=True, write_weekly=False)
        self.new_infected = new_infected
        S, E, I, R, H, V = history[-1]

        vaccines_left = self.vaccines_available - np.sum(decision)
        new_vaccines = np.sum(information['vaccine_supply'])

        vaccines_available = vaccines_left + new_vaccines

        return State(S, E, I, R, H, V, vaccines_available, time_step=self.time_step+decision_period)