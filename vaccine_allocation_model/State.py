import numpy as np

class State:
    def __init__(self, S, E, A, I, Q, R, D, V, vaccines_available, time_step):
        """ initialize a State instance

        Parameters
            S: array with shape (356,5) indicating number of suceptible in each region for each age group
            E: array with shape (356,5) indicating number of exposed in each region for each age group
            A: array with shape (356,5) indicating number of asymptomatic infected in each region for each age group
            I: array with shape (356,5) indicating number of symptomatic infected in each region for each age group
            Q: array with shape (356,5) indicating number of people in quarantine in each region for each age group
            R: array with shape (356,5) indicating number of recovered in each region for each age group
            D: array with shape (356,5) indicating number of accumulated deaths in each region for each age group
            V: array with shape (356,5) indicating number of vaccinated in each region for each age group
            vaccines_available: integer indicating number of vaccines available at initialization time step
            time_Step: integer indicating the time step when state is intialized in the range(0, (24/time_delta)*7 -1)
        Returns
            an initialized State instance

        """
        self.S = S
        self.E = E
        self.A = A
        self.I = I
        self.Q = Q
        self.R = R
        self.D = D
        self.V = V
        
        self.vaccines_available = vaccines_available
        self.time_step = time_step
        self.new_infected = 0


    def get_transition(self, decision, information, epidemic_function, decision_period):
        """ 
        Parameters
            decision: array indicating the number of vaccines to be allocated to each region for the decision period e.g (28, 356)
            information: dicitionary with exogenous information e.g {'vaccine_supply': (28, 356) }
            epidemic_function: function that simulated an epidemic 
            decision_period: the number of time steps that every decision directly affects
        Returns
            A new initialized State instance
        """
        # Update compartment values
        S, E, A, I, Q, R, D, V, new_infected = epidemic_function(self, decision, decision_period, information)
        self.new_infected = new_infected

        # Update vaccine available
        vaccines_left = self.vaccines_available - np.sum(decision)
        new_vaccines = np.sum(information['vaccine_supply'])
        vaccines_available = vaccines_left + new_vaccines

        # Update time step
        time_step=self.time_step+decision_period

        return State(S, E, A, I, Q, R, D, V, vaccines_available, time_step)
    

    def get_compartments_values(self):
        """ Returns compartment values """
        return [self.S, self.E, self.A, self.I, self.Q, self.R, self.D, self.V]