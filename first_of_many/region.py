from SIR import SIR
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
        self.sir_model = SIR()
    
    
    def model_outbreak(self):
        self.sir_model.outbreak(self.s0, self.i0, self.r0)
        self.sir_model.plot_outbreak()
    