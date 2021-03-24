import numpy as np

class State:
    def __init__(S, E, I, R, H, V, vaccines_available, time_step):
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.H = H
        self.V = V
        self.vaccines_available = vaccines_available
        self.time_step = time_step


    def get_transition(self, decision, information, epidemic_function, decision_period):
        vaccines_available = self.vaccines_available - np.sum(decision.astype(dtype='int64'), axis=0)
        try:
            vaccines_available += sum(information['vaccine_supply'][self.time_step:self.time_step+decision_period])
        except:
            vaccines_available += sum(information['vaccine_supply'][self.time_step:])
            
        _, history = epidemic_function(self, decision, decision_period, information)
        S, E, I, R, H, V = history[-1]

        return State(S, E, I, R, H, V, vaccines_available, time_step=self.time_step+decision_period)