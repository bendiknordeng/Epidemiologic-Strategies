from SEAIQR import Case

if __name__ == '__main__':
    C = 0.1
    M = 0
    alpha = 0.8
    beta = 1
    epsilon = 0.95
    mu = 0.0001
    gamma = 1/14
    sigma = 0.1
    omega = 0.1
    delta = 0.02
    p = 0.05

    case = Case(C=C, M=M, alpha=alpha, beta=beta, epsilon=epsilon, mu=mu, gamma=gamma, sigma=sigma, omega=omega, delta=delta, p=p)
    y = case.simulate_epidemic()
    case.plot_simulation(y)