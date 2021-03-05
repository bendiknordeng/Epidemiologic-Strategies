import random
import matplotlib.pyplot as plt
import numpy as np

class SIR:
    def __init__(self, beta=0.05, gamma=0.2) -> None:
        self.beta = beta
        self.gamma = gamma

    def outbreak(self, s0, i0, r0, T=200):
        S, I, R = s0, i0, r0
        Y= [(S,I,R)]
        for n in range(T):
            beta = 0.5*(I/(S+I+R))
            
            new_infected = np.random.binomial(S, beta)
            new_recovered = np.random.binomial(I, self.gamma)
            
            S,I,R = (S-new_infected, I+new_infected-new_recovered, R+new_recovered)
            Y.append((S,I,R))
        self.last_Y=Y

    def plot_outbreak(self):
        print("plot")
        plt.plot(self.last_Y)
        plt.ylabel('Number of individuals')
        plt.xlabel('Steps')
        plt.legend(('Susceptible', 'Infected', 'Recovered'))
        plt.show()


"""
    def transition(self, state, beta, gamma):
        prob = random.uniform(0,1)
        if state == 0:
            return prob <= beta
        elif state == 1:
            return prob <= gamma
        elif state == 2:
            return False

    def next_state(self, initial, N):
        transition_times = []
        for i in range(N):
            next_state = False
            counter = 0
            while not next_state:
                counter += 1
                next_state = self.transition(initial, self.beta, self.gamma)
            transition_times.append(counter)
        return sum(transition_times)/N
"""

