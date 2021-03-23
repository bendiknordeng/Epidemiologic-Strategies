import .covid.simulation as sim
import .covid.utils as utils
from state import State

class MarkovDecisionProcess:
    def __init__(self, config, horizon, vaccine_supply):
        self.horizon = horizon
        self.OD_matrices = sim.load_od_matrices(paths.od)
        self.pop, self.befolkning = utils.create_population(paths.muncipalities_names, paths.muncipalities_pop)
        self.vaccine_supply = utils.load_vaccination_programme(OD_matrices.shape[0], pop.shape[1], paths.municipalities_v)

        self.seir = SEIR(R0=config.R0,
                DE= config.DE* config.periods_per_day,
                DI= config.DI* config.periods_per_day,
                I0=initial,
                hospitalisation_rate=config.hospitalisation_rate,
                eff=config.eff,
                hospital_duration=config.hospital_duration*config.periods_per_day,
                time_delta=config.time_delta
                OD = self.OD_matrices, 
                pop = self.pop)

        self.state = State()
        self.path = [self.state]
        self.time = 0

    def run_policy(self, policy)
        for t in range(self.time, self.horizon, self.time_delta):
            decision = self.get_action(self.state, policy)
            information = self.get_exogenous_information(self.state)
            self.update_state(decision, information, 7)

    def get_action(self, state, policy):
        """ finds the action accoriding to a given policy at time t
        Parameters:
            state:
            policy: 
        Returns:
            number of vaccines to allocate to each of the municipalities at time t
        """
        return policies[policy](state)

    def get_exogenous_information(self, state):
        """ recieves the exogenous information at time t
        Parameters:
            t: time step
        Returns:
            returns a vector of alphas indicatinig the mobility flow at time t
        """
        alphas = [np.ones(self.OD_matrices.shape) for x in range(4)]
        information = {'alphas': alphas}
        return information
    
    def update_state(self, decision, information, days):
        """ recieves the exogenous information at time t
        Parameters:
            state: state
        Returns:
            returns a vector of alphas indicatinig the mobility flow at time t
        """
        res, history = self.seir.simulate(self.state, decision, self.vaccine_supply, information, days)
        S, E, I, R, H, V = history[-1]
        self.path.append(self.state)
        self.time += days
        self.state = State(S, E, I, R, H, V, self.time)

 


