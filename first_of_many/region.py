from SIR import SIR

class Region:
    def __init__(self, region_id, n_inhabitants, s0, i0, r0, longitude, latitude) -> None:
        # Demographics parameters
        self.region_id = region_id
        self.n_inhabitants = n_inhabitants
        
        # Epidemiological model parameters
        self.s0 = s0
        self.i0 = i0
        self.r0 = r0
        
        # Geographical parameters
        self.longitude = longitude
        self.latitude = latitude

        # Rates 
        sir_model = SIR()
        
        
if __name__ == '__main__':

    import os
    cwd = os.getcwd()
    print(cwd)   
