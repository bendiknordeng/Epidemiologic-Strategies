import numpy as np
from datetime import timedelta

class State:
    def __init__(self, S, E1, E2, A, I, R, D, V, contact_weights, alphas, vaccines_available,
                new_infected, total_infected, new_deaths, date, time_step=0):
        """ initialize a State instance

        Parameters
            S: array with shape (356,5) indicating number of suceptible in each region for each age group
            E1: array with shape (356,5) indicating number of latent exposed in each region for each age group
            E2: array with shape (356,5) indicating number of infectious exposed in each region for each age group
            A: array with shape (356,5) indicating number of asymptomatic infected in each region for each age group
            I: array with shape (356,5) indicating number of symptomatic infected in each region for each age group
            R: array with shape (356,5) indicating number of recovered in each region for each age group
            D: array with shape (356,5) indicating number of accumulated deaths in each region for each age group
            V: array with shape (356,5) indicating number of vaccinated in each region for each age group
            contact_weights: weights indicating weighting of contact matrices (Home, School, Work, Transport, Leisure)
            alphas, scaling factors indicating flow in compartments S, E1, E2, A and I
            vaccines_available: integer indicating number of vaccines available at initialization time step
            time_step: integer indicating the time step when state is intialized in the range(0, (24/time_delta)*7 -1)
        Returns
            an initialized State instance

        """
        self.S = S
        self.E1 = E1
        self.E2 = E2
        self.A = A
        self.I = I
        self.R = R
        self.D = D
        self.V = V
        
        self.contact_weights = np.array(contact_weights)
        self.alphas = np.array(alphas)
        self.vaccines_available = vaccines_available
        self.new_infected = new_infected
        self.total_infected = total_infected
        self.new_deaths = new_deaths
        self.date = date
        self.time_step = time_step

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
        S, E1, E2, A, I, R, D, V, new_infected, new_deaths = epidemic_function(self, decision, decision_period, information)

        # Update vaccine available
        vaccines_left = self.vaccines_available - np.sum(decision)
        new_vaccines = np.sum(information['vaccine_supply'])
        vaccines_available = vaccines_left + new_vaccines

        # Update time step
        time_step = self.time_step + decision_period
        date = self.date + timedelta(decision_period//4)

        # Update information
        contact_weights = information['contact_weights']
        alphas = information['alphas']

        return State(S, E1, E2, A, I, R, D, V, contact_weights, alphas, vaccines_available,
                    new_infected, self.total_infected+new_infected, new_deaths, date, time_step)
    

    def get_compartments_values(self):
        """ Returns compartment values """
        return [self.S, self.E1, self.E2, self.A, self.I, self.R, self.D, self.V]

    def __str__(self):
        total_pop = np.sum(self.get_compartments_values()[:-1])
        info = ["Susceptibles", "Exposed (latent)",
                "Exposed (presymptomatic)", "Asymptomatic infected",
                "Infected", "Recovered", "Dead", "Vaccinated", 
                "New infected", "Total infected"]
        values = [np.sum(compartment) for compartment in self.get_compartments_values()]
        values.append(np.sum(self.new_infected))
        values.append(np.sum(self.total_infected))
        percent = 100 * np.array(values)/total_pop
        status = f"Date: {self.date} (week {self.date.isocalendar()[1]})\n"
        status += f"Timestep: {self.time_step} (day {self.time_step//4})\n"
        for i in range(len(info)):
            status += f"{info[i]:<25} {values[i]:>7.0f} ({percent[i]:>5.2f}%)\n"
        return status

    @staticmethod
    def initialize_state(num_initial_infected, vaccines_available, contact_weights, alphas, population, start_date, time_step=0):
        """ Initializes a state, default from the moment a disease breaks out

        Parameters
            num_initial_infected: number of infected persons to be distributed randomly across regions if initiaL_infected=None e.g 50
            vaccines_available: int, number of vaccines available at time
            time_step: timestep in which state is initialized. Should be in the range of (0, (24/time_timedelta)*7 - 1)
        Returns
            an initialized State object, type defined in State.py
        """
        # pop = self.population.population.to_numpy(dtype='float64')
        pop = population[population.columns[2:-1]].to_numpy(dtype="float64")
        S = pop.copy()
        E1 = np.zeros(pop.shape)
        E2 = np.zeros(pop.shape)
        A = np.zeros(pop.shape)
        I = np.zeros(pop.shape)
        R = np.zeros(pop.shape)
        D = np.zeros(pop.shape)
        V = np.zeros(pop.shape)

        while num_initial_infected > 0:
            region = np.random.randint(0, pop.shape[0])
            age_group = np.random.randint(0, pop.shape[1])
            if S[region][age_group] > 0:
                num_initial_infected -= 1
                S[region][age_group] -= 1
                I[region][age_group] += 1 

        return State(S, E1, E2, A, I, R, D, V, contact_weights, alphas, vaccines_available, I.copy(), I.copy(), 0, start_date, time_step) 