import .covid.simulation as sim
import .covid.utils as utils
from state import State

class MarkovDecisionProcess:
    def __init__(self, config, horizon, time_delta, vaccine_supply):
        self.horizon = horizon
        self.time_delta = time_delta
        self.OD_matrices = sim.load_od_matrices(paths.od)
        self.pop, self.befolkning = utils.create_population(paths.muncipalities_names, paths.muncipalities_pop)
        self.seir = sim.initialize_seir(config, pop.shape[1], 50)
        self.vacc = utils.load_vaccination_programme(OD_matrices.shape[0], pop.shape[1], paths.municipalities_v)
        self.state = State()
        self.path = [self.state]
        self.vaccine_supply = vaccine_supply

    def run_policy(self, policy)
        for t in range(0, self.horizon, self.time_delta):


    def get_action(self, state, policy):
        """ finds the action accoriding to a given policy at time t
        Parameters:
            state:
            policy: 
        Returns:
            number of vaccines to allocate to each of the municipalities at time t
        """

    def get_exogenous_information(self):
        """ recieves the exogenous information at time t
        Parameters:
            t: time step
        Returns:
            returns a vector of alphas indicatinig the mobility flow at time t
        """

    
    def update_state(self, information, period=28):
        """ recieves the exogenous information at time t
        Parameters:
            state: state
        Returns:
            returns a vector of alphas indicatinig the mobility flow at time t
        """
        self.state = self.seir.simulate(self.state, information, period)




