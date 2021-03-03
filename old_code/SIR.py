import random
import matplotlib.pyplot as plt
import numpy as np


def transition(state, beta, gamma):
    prob = random.uniform(0,1)
    if state == 0:
        return prob <= beta
    elif state == 1:
        return prob <= gamma
    elif state == 2:
        return False

def next_state(initial, N, beta, gamma):
    transition_times = []
    for i in range(N):
        next_state = False
        counter = 0
        while not next_state:
            counter += 1
            next_state = transition(initial, beta, gamma)
        transition_times.append(counter)
    return sum(transition_times)/N


beta = 0.05
gamma = 0.2

print("Average transition time S -> I:", next_state(0,1000, beta, gamma))
print("Average transition time I -> R:", next_state(1,1000, beta, gamma))


def outbreak(T, Y_0, N, gamma):
    Y = [Y_0]
    for n in range(N):
        S, I, R = Y[n]
        beta = 0.5*(I/T)
        
        new_infected = np.random.binomial(S, beta)
        new_recovered = np.random.binomial(I, gamma)
        
        Y_next = (S-new_infected, I+new_infected-new_recovered, R+new_recovered)
        Y.append(Y_next)

    return Y

def plot_outbreak(Y):
    plt.plot(Y)
    plt.ylabel('Number of individuals')
    plt.xlabel('Steps')
    plt.legend(('Susceptible', 'Infected', 'Recovered'))
    plt.show()


T = 1000
Y_0 = (950, 50, 0)
N = 200
gamma = 0.2

Y = outbreak(T, Y_0, N, gamma)
plot_outbreak(Y)


T = 1000
Y_0 = (950, 50, 0)
N = 200
gamma = 0.2

steps_until_max = []
I_max = []
for i in range(1000):
    Y = outbreak(T, Y_0, N, gamma)
    I_n = [value[1] for value in Y]
    max_value = max(I_n)
    I_max.append(max_value)
    steps_until_max.append(I_n.index(max_value))

print('Expected maximum of infected people:', sum(I_max)/1000)
print('Expected time at maximum:', sum(steps_until_max)/1000)

