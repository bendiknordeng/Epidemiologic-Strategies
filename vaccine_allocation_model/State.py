import numpy as np

class State:
    def __init__(self, S, E, A, I, Q, R, D, V, H, vaccines_available, time_step, compartmets=None):
        """ initialize a State instance

        Parameters
            compartmets: {
                '0-65':{
                    S: array with shape (1,356) indicating number of suceptible in each region 
                    E: array with shape (1,356) indicating number of exposed in each region 
                    A: array with shape (1,356) indicating number of asymptomatic infected in each region 
                    I: array with shape (1,356) indicating number of symptomatic infected in each region 
                    Q: array with shape (1,356) indicating number of people in quarantine in each region 
                    R: array with shape (1,356) indicating number of recovered in each region 
                    D: array with shape (1,356) indicating number of accumulated deaths in each region 
                    V: array with shape (1,356) indicating number of vaccinated in each region
                    H: array with shape (1,356) indicating number of hospitalized in each region 
                    }, 
                '65+':{S:(1,356), E:(1,356), ...}}
            vaccines_available: integer indicating number of vaccines available at initialization time step
            time_Step: integer indicating the time step when state is intialized in the range(0, (24/time_delta)*7 -1)
        Returns
            an initialized State instance

        """
        self.compartmets = compartmets
        self.S = S
        self.E = E
        self.A = A
        self.I = I
        self.Q = Q
        self.R = R
        self.D = D
        self.V = V
        self.H = H
        
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

        # NEED TO UPDATE - delete SEAIQRDHV values initialization
        state_compartments_values = epidemic_function(self, decision, decision_period, information, hidden_cases=False, write_to_csv=True, write_weekly=False)
        _, new_infected, history = state_compartments_values['65+']
        self.new_infected = new_infected
        S, E, A, I, Q, R, D, V, H = history[-1]

        vaccines_left = self.vaccines_available - np.sum(decision)
        new_vaccines = np.sum(information['vaccine_supply'])

        vaccines_available = vaccines_left + new_vaccines
        time_step=self.time_step+decision_period

        return State(S, E, A, I, Q, R, D, V, H, vaccines_available, time_step, state_compartments_values)
    

    def get_compartments_values(self):
        """ Gets compartment values from a state
        Returns
            compartments    
        """
        return [self.S, self.E, self.A, self.I, self.Q, self.R, self.D, self.V, self.H]