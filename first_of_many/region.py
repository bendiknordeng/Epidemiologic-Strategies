#from SIR import SIR
from spread import SIR
import numpy as np

class Region:
    def __init__(self, region_id, n_inhabitants, initial_infected, longitude, latitude) -> None:
        # Demographics parameters
        self.region_id = region_id
        self.n_inhabitants = n_inhabitants
        
        # Epidemiological model parameters
        self.s0 = int(np.ceil(self.n_inhabitants*0.95))
        self.i0 = int(np.floor(self.n_inhabitants*0.05))
        self.r0 = 0
        
        # Geographical parameters
        self.longitude = longitude
        self.latitude = latitude

        # SIR-model
        kwargs = {"beta":1, 
            "p":0.05, 
            "sigma":0.1, 
            "time_steps":100, 
            "dt":1,
            "y0":[10000, 100, 0] # S0, I0, R0 
            }
     
        self.sir_model = SIR(**kwargs)
    
    
    def model_outbreak(self):
        self.sir_model
        self.sir_model.simulate_epidemic()
        self.sir_model.plot_simulation()
    